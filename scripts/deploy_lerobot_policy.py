"""Deploy LeRobot policy with isolated inference in a separate process."""

import os
import argparse
import shutil
import time
from pathlib import Path
from multiprocessing import Process, Pipe

import torch      
# torch.backends.cudnn.benchmark = True     # Probably not needed

import numpy as np
from rich import print

from crisp_py.gripper import GripperConfig

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from crisp_gym.lerobot_wrapper import get_features
from crisp_gym.manipulator_env import ManipulatorCartesianEnv
from crisp_gym.manipulator_env_config import AlohaFrankaEnvConfig
from crisp_gym.record.recording_manager import KeyboardRecordingManager


def inference_worker(conn, policy_path, n_action_steps, n_inference_steps, task_str, img_shape, env):
    """
    Policy inference process: loads policy on GPU, receives observations via conn,
    returns actions, and exits on None.
    """
    device = torch.device('cuda')
    # Load and configure policy
    if "dp" in str(policy_path):
        policy = DiffusionPolicy.from_pretrained(policy_path)
        policy.config.num_inference_steps = n_inference_steps
    else:
        policy = SmolVLAPolicy.from_pretrained(policy_path)
    policy.config.n_action_steps = n_action_steps
    policy.reset()
    policy.to(device).eval()

    # # Warm up CUDA kernels
    warmup_obs = {
        'observation.state': torch.zeros((1, 7), device=device)
    }
    for cam in env.cameras:
        warmup_obs[f"observation.images.{cam.config.camera_name}"] = torch.zeros((1, 3, *img_shape), device=device)

    # warmup_obs["observation.images.right_third_person_camera"] = warmup_obs["observation.images.primary"]
    # warmup_obs["observation.images.right_wrist_camera"] = warmup_obs["observation.images.wrist"]
    warmup_obs['task'] = task_str
    
    with torch.inference_mode():
        _ = policy.select_action(warmup_obs)
        torch.cuda.synchronize()
    print('[Inference] Warm-up complete')

    # Main inference loop
    while True:
        obs = conn.recv()
        if obs is None:
            break
        if obs == "reset":
            print("[Inference] Resetting policy")
            policy.reset()
            continue
        with torch.inference_mode():
            action = policy.select_action(obs)
        print(f"Computed following action: {action} at time {time.time()}")
        conn.send(action)

    conn.close()
    print('[Inference] Worker shutting down')

def main():
    parser = argparse.ArgumentParser(description="Deploy LeRobot policy")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="franka_single",
        help="Repository ID for the dataset",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="green",
        help="Task description",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="franka",
        help="Type of robot being used",
    )
    parser.add_argument(
        "--policy-path", 
        type=str, 
        default="/home/ralf/Projects/lerobot/outputs/train/smolvla_pick_green_lego_block_filtered_new/checkpoints/020000/pretrained_model", 
        help="Path for the policy checkpoint"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for recording",
    )
    parser.add_argument(
        "--action_execution_steps",
        type=int,
        default=10,
        help="Number of actions to execute before replanning",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum episode duration in seconds",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Number of episodes to record",
    )
    parser.add_argument(
        "--resume",
        type=bool,
        default=False,
        help="Resume recording of an already existing dataset",
    )
    parser.add_argument(
        "--push-to-hub",
        type=bool,
        default=False,
        help="Whether to push the dataset to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--leader-controller",
        type=str,
        default="gravity_compensation_on_plane",
        help="Controller configuration for the leader robot.",
    )
    parser.add_argument(
        "--right-gripper-config",
        type=str,
        default="gripper_right",
        help="Gripper configuration for the right robot.",
    )
    parser.add_argument(
        "--time-to-home",
        type=float,
        default=3.0,
        help="Time needed to home.",
    )
    parser.add_argument(
        "--trigger-config",
        type=str,
        default="trigger",
        help="Trigger configuration for the leader robot.",
    )
    # TODO: @maxdoesch add this
    parser.add_argument(
        "--joint-control",
        type=bool,
        default=False,
        help="Whether to use joint control for the robot.",
    )
    
    args = parser.parse_args()

    home_config = [
        -1.73960110e-02,
        9.55319758e-02,
        8.09703053e-04,
        -1.94272034e00,
        -4.01435784e-03,
        2.06584183e00,
        7.97426445e-01,
    ]

    # Clean existing local dataset
    if Path(HF_LEROBOT_HOME / args.repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / args.repo_id)

    # Set up the config for the environment
    path_to_config = os.environ.get("CRISP_CONFIG_PATH")
    if path_to_config is None:
        raise ValueError(
            "You need to set the environment variable CRISP_CONFIG_PATH in order to load configs for the gripper and controller.\nTo do this execute export CRISP_CONFIG_PATH=path\\to\\config."
        )

    # Set up the envionment configuration
    gripper_config = GripperConfig.from_yaml(
        path=(Path(path_to_config) / (args.right_gripper_config + ".yaml")).resolve()
    )
    gripper_config.joint_state_topic = "gripper" + "/" + gripper_config.joint_state_topic
    gripper_config.command_topic = "gripper" + "/" + gripper_config.command_topic

    env_config = AlohaFrankaEnvConfig(gripper_config=gripper_config)
    env_config.robot_config.home_config = home_config
    env_config.robot_config.time_to_home = args.time_to_home
    env = ManipulatorCartesianEnv(namespace="right", config=env_config)

    # %% Prepare the dataset
    features = get_features(env)

    if args.resume:
        print(f"[green]Resuming recording from existing dataset: {args.repo_id}")
        dataset = LeRobotDataset(repo_id=args.repo_id)
    else:
        print(f"[green]Creating new dataset: {args.repo_id}")
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            robot_type=args.robot_type,
            features=features,
            use_videos=True,
        )

    # %% Prepare environment and leader
    env.home()
    env.reset()

    # Setup multiprocessing Pipe
    parent_conn, child_conn = Pipe()
    # Determine image shape for warmup
    img_shape = features['observation.images.primary']['shape'][-2:]

    # Start inference process
    inf_proc = Process(
        target=inference_worker,
        args=(child_conn, args.policy_path, args.action_execution_steps, 20, args.task, img_shape, env),
        daemon=True,
    )
    inf_proc.start()

    time.sleep(1.0)  # Give some time for the process to start

    # Recording and main control loop
    with KeyboardRecordingManager(num_episodes=args.num_episodes) as rm:
        while not rm == "exit":
            print(
                f"[magenta bold]=== Episode {rm.episode_count + 1} / {rm.num_episodes} ==="
            )   

            if rm.state == "is_waiting":
                print("[magenta]Waiting for user to start.")
                while rm.state == "is_waiting":
                    time.sleep(1.0)

            print("[blue]Started episode")

            obs = {}

            env.home()
            env.reset()

            # Reset the policy
            parent_conn.send("reset")

            while rm.state == "recording":
                step_time_init = time.time()

                obs_raw = env._get_obs()
                # Build observation tensors
                state = np.concatenate([obs_raw['cartesian'][:6], obs_raw['gripper']])

                # Hack: DELETE
                # if 'nostates' in args.policy_path:
                #     state[:6] = 0.0
                # elif 'filtered_new' in args.policy_path:
                #     state[[3, 4]] = 0.0

                obs = {
                    'observation.state': torch.from_numpy(state).unsqueeze(0).cuda().float()
                }
                for cam in env.cameras:
                    img = obs_raw[f"{cam.config.camera_name}_image"]
                    obs[f"observation.images.{cam.config.camera_name}"] = \
                        torch.from_numpy(img).permute(2,0,1).unsqueeze(0).cuda().float() / 255
                obs['task'] = args.task

                # obs["observation.images.right_third_person_camera"] = obs["observation.images.primary"]
                # obs["observation.images.right_wrist_camera"] = obs["observation.images.wrist"]

                # Offload inference
                parent_conn.send(obs)
                action = parent_conn.recv().squeeze(0).cpu().numpy()

                # Clip the gripper action to be between 0 and 1
                # action[6] = np.clip(action[6], 0.0, 1.0)  # Ensure gripper action is between 0 and 1
                # if action[6] < 0.5:
                #     action[6] = min(action[6], 0.3)
                # action_safe = np.array([0.0, 0.0, -1e-3, 0.0, 0.0, 0.0, 1.0])  # Default action if policy fails

                # Step environment and record
                env.step(action, block=False)
                action_dict = {dim: action[i] for i, dim in enumerate(features["action"]["names"])}
                obs_dict = {
                    dim: obs_raw["cartesian"][i] if i < 6 else obs_raw["gripper"][0]
                    for i, dim in enumerate(features["observation.state"]["names"])
                }
                cam_frame = {
                    f"observation.images.{camera.config.camera_name}": obs_raw[
                        f"{camera.config.camera_name}_image"
                    ]
                    for camera in env.cameras
                }
                action_frame = build_dataset_frame(features, action_dict, prefix="action")
                obs_frame = build_dataset_frame(features, obs_dict, prefix="observation.state")

                frame = {**obs_frame, **action_frame, **cam_frame}
                dataset.add_frame(frame, task=args.task)

                sleep_time = 1 / args.fps - (time.time() - step_time_init)
                if sleep_time > 0:
                    time.sleep(sleep_time)  # Sleep to allow the environment to process the action

            if rm.state == "paused":
                print(
                    "[blue] Stopped episode. Waiting for user to decide whether to save or delete the episode"
                )
                env.robot.home(blocking=False)
            
            while rm.state == "paused":
                time.sleep(1.0)

            if rm.state == "to_be_saved":
                print("[green]Saving episode.")
                dataset.save_episode()
                rm.episode_count += 1
                rm.set_to_wait()

            if rm.state == "to_be_deleted":
                print("[red]Deleting episode")
                dataset.clear_episode_buffer()
                rm.set_to_wait()

            if rm.state == "exit":
                break

    # Shutdown inference process
    parent_conn.send(None)
    inf_proc.join()

    if args.push_to_hub:
        print(f"[green]Pushing dataset to Hugging Face Hub with repo_id: {args.repo_id}")
        dataset.push_to_hub(repo_id=args.repo_id, private=False)

    env.home()
    env.close()

if __name__ == '__main__':
    main()
