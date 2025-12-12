"""Script showcasing how to record data in Lerobot Format."""

import argparse
from dataclasses import dataclass
import datetime
import logging
from pathlib import Path
from typing import Literal

import crisp_gym  # noqa: F401
from crisp_gym.envs.manipulator_env import make_env
from crisp_gym.envs.manipulator_env_config import list_env_configs
from crisp_gym.policy import make_policy
from crisp_gym.policy.policy import list_policy_configs
from crisp_gym.record.evaluate import Evaluator
from crisp_gym.record.recording_manager import make_recording_manager
from crisp_gym.util import prompt
from crisp_gym.util.lerobot_features import get_features
from crisp_gym.util.setup_logger import setup_logging


@dataclass
class DeploymentArgs:
    """Arguments for deploying a pretrained policy."""

    repo_id: str | None = None

    env_namespace: str | None = None
    env_config: str | None = None
    policy_repo_id_or_path: str | None = None
    policy_config: str | None = None

    robot_type: str = "franka"
    fps: int = 15
    num_episodes: int = 10
    resume: bool = False
    push_to_hub: bool = False
    recording_manager_type: Literal["ros", "keyboard"] = "keyboard"
    joint_control: bool = False
    log_level: str = "INFO"
    evaluate: bool = False
    task: str = "Finish the task."
    peft_weight_path: Path | None = None

    @classmethod
    def from_argparser(cls) -> "DeploymentArgs":
        """Create DeploymentArgs from argparse.Namespace."""
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
            default=False,
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
            "--policy-repo-id-or-path",
            type=str,
            default=None,
            help="Path to the pretrained model or repo id of the policy (if not provided, a prompt will ask you to select one from 'outputs/train')",
        )
        parser.add_argument(
            "--env-config",
            type=str,
            default=None,
            help="Configuration name for the follower robot. You can define your own configurations, please check https://utiasdsl.github.io/crisp_controllers/misc/create_own_config/.",
        )
        parser.add_argument(
            "--policy-config",
            type=str,
            default=None,
            help="Path to a custom policy configuration file (YAML). If not provided, the default configuration for the selected policy will be used.",
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
        parser.add_argument(
            "--task",
            type=str,
            default="Finish the task",
            help="Language task command",
        )
        parser.add_argument(
            "--peft_weight_path",
            type=str,
            required=True,
            help="Path of peft adapter.",
        )
        args = parser.parse_args()
        return cls(**vars(args))

    @classmethod
    def from_yaml(cls, yaml_path_str: str) -> "DeploymentArgs":
        """Create DeploymentArgs from a YAML file.

        Args:
            yaml_path_str (str): Path to the YAML file.

        Returns:
            DeploymentArgs: Deployment arguments.
        """
        import yaml

        yaml_path = Path(yaml_path_str)
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_path_str}")

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


def deploy_policy(args: DeploymentArgs):
    """Deploy a pretrained policy and record deployment data in Lerobot Format.

    Args:
        args (DeploymentArgs): Arguments for deployment.
    """
    logger = logging.getLogger(__name__)
    setup_logging(level=logging._nameToLevel[args.log_level])

    logger.info("-" * 40)
    logger.info("Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg:<30}: {value}")
    logger.info("-" * 40)

    if args.repo_id is None:
        args.repo_id = prompt.prompt(
            "Please enter the repository ID for the dataset (e.g., 'username/dataset_name'):",
        )
        logger.info(f"Using repository ID: {args.repo_id}")

    if args.policy_repo_id_or_path is None:
        logger.info(" No path provided. Searching for models in 'outputs/train' directory.")

        # We check recursively in the 'outputs/train' directory for 'pretrained_model's recursively
        models_path = Path("outputs/train")
        if models_path.exists() and models_path.is_dir():
            models = [model for model in models_path.glob("**/pretrained_model") if model.is_dir()]
            models_names = sorted([str(model) for model in models], key=lambda x: x.lower())

            args.policy_repo_id_or_path = prompt.prompt(
                message="Please select a model to use for deployment:",
                options=models_names,
                default=models_names[0] if models else None,
            )
            logger.info(f"Using policy repo id or path: {args.policy_repo_id_or_path}")
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
    if args.policy_config is None:
        policy_configs = list_policy_configs()
        args.policy_config = prompt.prompt(
            "Please select the policy configuration to use.",
            options=policy_configs,
            default=policy_configs[0],
        )

    if args.evaluate:
        logger.info("Evaluation mode enabled. Will evaluate the performance after each episode.")
        datetime_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        evaluation_file = (
            f"evaluation_results_{args.policy_repo_id_or_path.replace('/', '_')}_{datetime_now}.csv"
        )
    else:
        evaluation_file = "evaluation_results.csv"

    policy = None
    try:
        ctrl_type = "cartesian" if not args.joint_control else "joint"
        env = make_env(args.env_config, control_type=ctrl_type, namespace=args.env_namespace)
        env.task = args.task

        # %% Prepare the dataset
        features = get_features(env)

        evaluator = Evaluator(output_file="eval/" + evaluation_file)

        

        # %% Set up multiprocessing for policy inference
        logger.info("Setting up the policy.")
        policy = make_policy(
            name_or_config_name=args.policy_config,
            pretrained_path=args.policy_repo_id_or_path,
            env=env,
            peft_weight_path=args.peft_weight_path
        )    

        recording_manager = make_recording_manager(
            recording_manager_type=args.recording_manager_type,
            features=features,
            **{
                "repo_id": args.repo_id,
                "robot_type": args.robot_type,
                "num_episodes": args.num_episodes,
                "fps": args.fps,
                "resume": args.resume,
            },
        )
        recording_manager.wait_until_ready()


        logger.info("Homing robot before starting with recording.")

        env.home()
        env.reset()

        def on_start():
            """Hook function to be called when starting a new episode."""
            env.reset()
            policy.reset()
            evaluator.start_timer()

        def on_end():
            """Hook function to be called when stopping the recording."""
            env.robot.reset_targets()
            env.robot.home(blocking=False)
            env.gripper.open()

            logger.info("Waiting for user to decide on success/failure if evaluating...")
            if recording_manager.state != "exit":
                evaluator.evaluate(episode=recording_manager.episode_count)

        with evaluator.start_eval(overwrite=True, activate=args.evaluate):
            with recording_manager:
                while not recording_manager.done():
                    logger.info(
                        f"→ Episode {recording_manager.episode_count + 1} / {recording_manager.num_episodes}"
                    )

                    recording_manager.record_episode(
                        data_fn=policy.make_data_fn(),
                        task=args.task,
                        on_start=on_start,
                        on_end=on_end,
                    )

                    logger.info("Episode finished.")

        # Shutdown inference process
        logger.info("Shutting down inference process.")
        policy.shutdown()

        logger.info("Homing robot.")
        env.home()

        logger.info("Closing the environment.")
        env.close()

        logger.info("Finished recording.")
    except Exception as e:
        logger.exception(e)
        if policy is not None:
            policy.shutdown()


def main():
    """Main function to deploy a pretrained policy."""
    args = DeploymentArgs.from_argparser()
    deploy_policy(args)


if __name__ == "__main__":
    main()
