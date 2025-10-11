"""Converts multiple MCAP files to a single LeRobot Dataset."""

import argparse
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from rich.progress import track

from crisp_gym.manipulator_env import ManipulatorBaseEnv, make_env
from crisp_gym.manipulator_env_config import list_env_configs
from crisp_gym.record.mcap.episode_convert import (
    convert_mcap_file_to_lerobot_episode,
    get_features,
    get_fps_from_recording,
)
from crisp_gym.util.prompt import prompt
from crisp_gym.util.setup_logger import setup_logging

setup_logging()


def convert_multiple_mcap_files_to_lerobot_dataset(
    mcap_folder: Path | str,
    env: ManipulatorBaseEnv,
    dataset: LeRobotDataset,
):
    """Convert multiple MCAP files in a folder to a single LeRobot Dataset."""
    mcap_folder = Path(mcap_folder)
    mcap_files = list(mcap_folder.glob("*.mcap"))
    mcap_files.sort()

    for mcap_file in track(mcap_files, description=f"Converting MCAP files from {mcap_folder}"):
        convert_mcap_file_to_lerobot_episode(env=env, dataset=dataset, mcap_file=mcap_file)


if __name__ == "__main__":
    from rich.traceback import install

    install()

    parser = argparse.ArgumentParser(
        description="Convert multiple MCAP files in a folder to a single LeRobot Dataset and upload to Hugging Face Hub."
    )
    parser.add_argument(
        "--mcap_folder",
        type=str,
        required=True,
        help="Path to the folder containing MCAP files.",
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
        help="FPS of the dataset. If not provided, it will be estimated from the first MCAP file in the folder.",
    )
    args = parser.parse_args()

    mcap_folder = Path(args.mcap_folder)
    repo_id = args.repo_id
    fps = args.fps

    env_name = args.env_name
    namespace = args.namespace

    if args.env_name is None:
        env_name = prompt(options=list_env_configs())
    if args.namespace is None:
        namespace = prompt("Enter namespace (or leave blank for none): ")

    if fps is None:
        first_mcap_file = next(mcap_folder.glob("*.mcap"), None)
        if first_mcap_file is None:
            raise ValueError(f"No MCAP files found in folder: {mcap_folder}")
        fps = get_fps_from_recording(first_mcap_file)
        print(f"Estimated FPS from first recording: {fps}")

    env = make_env(env_name, namespace=namespace)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        features=get_features(env, ignore_keys=["observation.state", "observation.state.target"]),
        fps=fps,
        use_videos=True,
    )

    try:
        convert_multiple_mcap_files_to_lerobot_dataset(
            mcap_folder=mcap_folder,
            env=env,
            dataset=dataset,
        )
        dataset.push_to_hub()
    except Exception as e:
        env.close()
        raise e

    env.close()
