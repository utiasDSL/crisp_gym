"""Utils to convert MCAP files in a folder to a full LeRobot Dataset."""

import argparse
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from rich.progress import track

from crisp_gym.manipulator_env import ManipulatorBaseEnv, make_env
from crisp_gym.manipulator_env_config import list_env_configs
from crisp_gym.record.mcap.convert_episode import (
    convert_mcap_file_to_lerobot_episode,
    get_fps_from_recording,
)
from crisp_gym.util.lerobot_features import get_features
from crisp_gym.util.prompt import prompt
from crisp_gym.util.setup_logger import setup_logging

setup_logging()


def convert_mcap_folder_to_lerobot_dataset(
    env: ManipulatorBaseEnv,
    mcap_folder: Path | str,
    dataset: LeRobotDataset,
):
    """Convert an MCAP file to LeRobot Dataset format and upload to Hugging Face Hub.

    Args:
        env (ManipulatorBaseEnv): Environment to use for defining features.
        mcap_folder (Path | str): Path to the MCAP folder.
        dataset (LeRobotDataset): LeRobotDataset instance to add episodes to.
    """
    mcap_folder = Path(mcap_folder)
    assert mcap_folder.exists(), f"Folder with mcap files {mcap_folder} does not exist."
    mcap_files = list(mcap_folder.glob("*/*.mcap"))
    assert len(mcap_files) > 0, f"No MCAP files found in folder {mcap_folder}."
    print(f"Found {len(mcap_files)} MCAP files in folder {mcap_folder}.")
    for mcap_file in track(
        mcap_files, description=f"Converting MCAP files in {mcap_folder} to LeRobot Dataset..."
    ):
        convert_mcap_file_to_lerobot_episode(env=env, mcap_file=mcap_file, dataset=dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert an MCAP file to LeRobot Dataset format and upload to Hugging Face Hub."
    )
    parser.add_argument(
        "--mcap_folder",
        type=str,
        required=True,
        help="Path to the MCAP file.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face Hub repository ID to upload the dataset to.",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        help="Name of the environment to use.",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        help="Namespace for the environment (if applicable).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="FPS of the dataset. If not provided, it will be estimated from the /action topic in the MCAP file.",
    )
    args = parser.parse_args()

    mcap_folder = Path(args.mcap_folder)
    assert mcap_folder.exists(), f"Folder with mcap files {mcap_folder} does not exist."
    mcap_files = list(mcap_folder.glob("*/*.mcap"))
    assert len(mcap_files) > 0, f"No MCAP files found in folder {mcap_folder}."

    repo_id = args.repo_id
    fps = args.fps or get_fps_from_recording(mcap_file=mcap_files[0])

    env_name = args.env_name
    if args.env_name is None:
        env_configs = list_env_configs()
        env_name = prompt(
            options=env_configs,
            default=env_configs[0],
        )
    namespace = args.namespace
    if args.namespace is None:
        namespace = prompt(
            options=["right", "left"],
            default="right",
            message="Select the namespace for the environment:",
        )

    env = make_env(env_name, namespace=namespace)
    dataset = None

    try:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            features=get_features(env, ignore_keys=["observation.state"]),
            fps=fps,
            use_videos=True,
        )

        convert_mcap_folder_to_lerobot_dataset(env=env, mcap_folder=mcap_folder, dataset=dataset)
    except Exception as e:
        print(f"Error during conversion: {e}")
        env.close()
        if dataset is not None:
            dataset.push_to_hub()
