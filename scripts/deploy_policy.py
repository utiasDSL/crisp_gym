"""Script showcasing how to deploy a trained policy."""

import argparse
import time

import torch
from crisp_py.camera import FrankaCameraConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

from crisp_gym.manipulator_env import ManipulatorCartesianEnv
from crisp_gym.manipulator_env_config import OnlyWristCamFrankaEnvConfig
from crisp_gym.record.recording_manager import RecordingManager

parser = argparse.ArgumentParser(description="Evaluate a policy trained using LeRobot")


parser.add_argument("--num-episodes", type=int, default=3, help="Number of episodes to evaluate")
args = parser.parse_args()
num_episodes = args.num_episodes

# Select your device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create robot environment
camera_config = FrankaCameraConfig()
camera_config.camera_name = "right_wrist_camera"
camera_config.camera_color_image_topic = "right_wrist_camera/color/image_rect_raw"
camera_config.camera_color_info_topic = "right_wrist_camera/color/camera_info"

env_config = OnlyWristCamFrankaEnvConfig(camera_configs=[camera_config])
env = ManipulatorCartesianEnv(namespace="right", config=env_config)

# Load policy
policy = DiffusionPolicy.from_pretrained("lerobot/diffusion_policy_franka_single_v2")


# Compare environment and policy observation and action spaces
print(policy.config.input_features)
print(env.observation_space)

print(policy.config.output_features)
print(env.action_space)

# env.reset()
# env.robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")

policy.reset()
with RecordingManager(num_episodes=num_episodes) as recording_manager:
    while not recording_manager == "exit":
        print(
            f"[magenta bold]=== Episode {recording_manager.episode_count + 1} / {recording_manager.num_episodes} ==="
        )

        if recording_manager.state == "is_waiting":
            print("[magenta]Waiting for user to start.")
            while recording_manager.state == "is_waiting":
                time.sleep(1.0)

        print("[blue]Started episode")

        while recording_manager.state == "recording":
            # Get and process current observation
            # obs_pre_step = env._get_obs()

            # state = torch.from_numpy(
            #     np.concatenate(
            #     (
            #         obs_pre_step["cartesian"],
            #         np.array([obs_pre_step["gripper"]]),
            #     ),
            #     axis=0,
            # ))
            # cam_frame = {
            #     f"observation.images.{camera.config.camera_name}": obs_pre_step[
            #         f"{camera.config.camera_name}_image"
            #     ]
            #     for camera in env.cameras
            # }
            # # Stack images from all cameras
            # image = torch.from_numpy(
            #     np.stack(
            #         [obs_pre_step[f"{camera.config.camera_name}_image"] for camera in env.cameras],
            #         axis=0,
            #     )
            # )

            # state = state.to(torch.float32)
            # image = image.to(torch.float32) / 255
            # image = image.permute(2, 0, 1)

            # # Send data tensors to device and unsqueeze to add batch dimension
            # state = state.to(device, non_blocking=True).unsqueeze(0)
            # image = image.to(device, non_blocking=True).unsqueeze(0)

            # # Create the policy input dictionary
            # observation = {
            #     "observation.state": state,
            #     "observation.image": image,
            # }

            # # Compute action
            # with torch.inference_mode():
            #     action = policy.select_action(obs_pre_step)
            # numpy_action = action.squeeze(0).to("cpu").numpy()

            # # Step the environment with the action
            # obs_after_step, _, _, _, _ = env.step(numpy_action)

            # action = np.concatenate(
            #     (
            #         obs_after_step["cartesian"],
            #         np.array([obs_after_step["gripper"]]),
            #     ),
            #     axis=0,
            # )
            pass

        if recording_manager.state == "paused":
            print(
                "[blue] Stopped episode. Waiting for user to decide whether to save or delete the episode"
            )
        while recording_manager.state == "paused":
            time.sleep(1.0)

        if recording_manager.state == "to_be_saved":
            print("[green]Saving episode.")
            # TODO: Implement saving logic here
            recording_manager.episode_count += 1
            env.home()
            policy.reset()
            recording_manager.set_to_wait()

        if recording_manager.state == "to_be_deleted":
            print("[red]Deleting episode")
            env.home()
            policy.reset()
            recording_manager.set_to_wait()

        if recording_manager.state == "exit":
            break


# env.home()
# env.close()
