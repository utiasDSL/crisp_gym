"""Script showcasing how to record data in Lerobot Format."""

import argparse
import datetime
import logging
import time
from multiprocessing import Pipe, Process

import crisp_gym  # noqa: F401
from crisp_gym.config.home import home_close_to_table
from crisp_gym.manipulator_env import make_env
from crisp_gym.manipulator_env_config import list_env_configs
from crisp_gym.record.evaluate import Evaluator
from crisp_gym.record.record_functions import inference_worker, make_policy_fn
from crisp_gym.record.recording_manager import make_recording_manager
from crisp_gym.util import prompt
from crisp_gym.util.lerobot_features import get_features
from crisp_gym.util.setup_logger import setup_logging


def main():
    """Deploy a pretrained policy and record deployment data in Lerobot Format."""
    parser = argparse.ArgumentParser(
        description="Deploy a pretrained policy and record data in Lerobot Format"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Repository ID for the dataset",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="franka",
        help="Type of robot being used.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Frames per second for recording",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to record",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume recording of an already existing dataset",
    )
    parser.add_argument(
        "--push-to-hub",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to push the dataset to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--recording-manager-type",
        type=str,
        default="keyboard",
        help="Type of recording manager to use. Currently only 'keyboard' and 'ros' are supported.",
    )
    parser.add_argument(
        "--joint-control",
        action="store_true",
        help="Whether to use joint control for the robot.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logger level.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to the pretrained model (if not provided, a prompt will ask you to select one from 'outputs/train')",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default=None,
        help="Configuration name for the follower robot. Define your own configuration in `crisp_gym/manipulator_env_config.py`.",
    )
    parser.add_argument(
        "--env-namespace",
        type=str,
        default=None,
        help="Namespace for the follower robot. This is used to identify the robot in the ROS ecosystem.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="Whether to evaluate the performance of the model after each episode.",
    )

    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    setup_logging(level=args.log_level)

    logger.info("-" * 40)
    logger.info("Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    logger.info("-" * 40)

    if args.repo_id is None:
        args.repo_id = prompt.prompt(
            "Please enter the repository ID for the dataset (e.g., 'username/dataset_name'):",
        )
        logger.info(f"Using repository ID: {args.repo_id}")

    if args.path is None:
        logger.info(" No path provided. Searching for models in 'outputs/train' directory.")
        from pathlib import Path

        # We check recursively in the 'outputs/train' directory for 'pretrained_model's recursively
        models_path = Path("outputs/train")
        if models_path.exists() and models_path.is_dir():
            models = [model for model in models_path.glob("**/pretrained_model") if model.is_dir()]
            models_names = sorted([str(model) for model in models], key=lambda x: x.lower())

            args.path = prompt.prompt(
                message="Please select a model to use for deployment:",
                options=models_names,
                default=models_names[0] if models else None,
            )
            logger.info(f"Using model path: {args.path}")
        else:
            logger.error("'outputs/models' directory does not exist.")
            logger.error(
                "Please provide a valid path to the model using --path or create a new one."
            )
            exit(1)

    if args.env_namespace is None:
        args.env_namespace = prompt.prompt(
            "Please enter the follower robot namespace (e.g., 'left', 'right', ...)",
            default="right",
        )
        logger.info(f"Using follower namespace: {args.env_namespace}")

    if args.env_config is None:
        follower_configs = list_env_configs()
        args.env_config = prompt.prompt(
            "Please enter the follower robot configuration name.",
            options=follower_configs,
            default=follower_configs[0],
        )
        logger.info(f"Using follower configuration: {args.env_config}")

    if args.evaluate:
        logger.info("Evaluation mode enabled. Will evaluate the performance after each episode.")
        datetime_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        evaluation_file = (
            prompt.prompt(
                "Please enter the output file for evaluation results",
                default=f"evaluation_results_{args.path.replace('/', '_')}_{datetime_now}",
            )
            + ".csv"
        )
    else:
        evaluation_file = "evaluation_results.csv"

    inf_proc = None
    try:
        ctrl_type = "cartesian" if not args.joint_control else "joint"
        env = make_env(args.env_config, control_type=ctrl_type, namespace=args.env_namespace)
        # TODO: Remove this hack!
        env.config.robot_config.home_config = home_close_to_table
        env.config.robot_config.time_to_home = 2.0

        # %% Prepare the dataset
        features = get_features(env)

        evaluator = Evaluator(output_file="eval/" + evaluation_file)

        recording_manager = make_recording_manager(
            recording_manager_type=args.recording_manager_type,
            features=features,
            repo_id=args.repo_id,
            robot_type=args.robot_type,
            num_episodes=args.num_episodes,
            fps=args.fps,
            resume=args.resume,
        )
        recording_manager.wait_until_ready()

        # %% Set up multiprocessing for policy inference
        logger.info("Setting up multiprocessing for policy inference.")
        parent_conn, child_conn = Pipe()

        # Start inference process
        inf_proc = Process(
            target=inference_worker,
            kwargs={
                "conn": child_conn,
                "pretrained_path": args.path,
                "env": env,
            },
            daemon=True,
        )
        inf_proc.start()

        time.sleep(5.0)  # Give some time for the process to start

        logger.info("Homing robot before starting with recording.")

        env.home()
        env.reset()

        def on_start():
            """Hook function to be called when starting a new episode."""
            env.reset()
            parent_conn.send("reset")
            evaluator.start_timer()

        def on_end():
            """Hook function to be called when stopping the recording."""
            env.robot.reset_targets()
            env.robot.home(blocking=False)
            env.gripper.open()

            logger.info("Waiting for user to decide on success/failure if evaluating...")
            time.sleep(3.0)
            if recording_manager.state != "exit":
                evaluator.evaluate(episode=recording_manager.episode_count)

        with evaluator.start_eval(overwrite=True, activate=args.evaluate):
            with recording_manager:
                while not recording_manager.done():
                    logger.info(
                        f"→ Episode {recording_manager.episode_count + 1} / {recording_manager.num_episodes}"
                    )

                    recording_manager.record_episode(
                        data_fn=make_policy_fn(env, parent_conn),
                        task="Pick up the lego block.",
                        on_start=on_start,
                        on_end=on_end,
                    )

                    logger.info("Episode finished.")

        # Shutdown inference process
        logger.info("Shutting down inference process.")
        parent_conn.send(None)
        inf_proc.join()

        logger.info("Homing robot.")
        env.home()

        logger.info("Closing the environment.")
        env.close()

        logger.info("Finished recording.")
    except Exception as e:
        logger.exception(e)
        if inf_proc is not None and inf_proc.is_alive():
            logger.info("Shutting down inference process due to exception.")
            inf_proc.join()


if __name__ == "__main__":
    main()
