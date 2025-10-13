"""Utils to convert MCAP files to a LeRobot Dataset episode.

MCAP files are a format for storing ROS2 messages and other time-series data.
MCAP stores much more information that LeRobotDataset, since it consists of raw measurements that are stored in multiple
frequencies (with timestamps and more), while LeRobotDataset stores synchronized data at a fixed frequency.

There are multiple ways to convert MCAP files to LeRobotDataset format, depending on the use case.
We simply use the an environment defined in crisp_gym to define the features and their frequencies.
"""

import argparse
from pathlib import Path

import numpy as np
from crisp_py.robot import Pose
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from mcap.reader import NonSeekingReader
from mcap_ros2.reader import read_ros2_messages
from rich import print
from rich.progress import track

from crisp_gym.manipulator_env import ManipulatorBaseEnv, make_env
from crisp_gym.util.lerobot_features import get_features


def combine_actions(actions: list[np.ndarray], axis: int = 0) -> np.ndarray:
    """Combine multiple action arrays into a single action array."""
    action_pose = np.zeros((6,), dtype=np.float32)
    action_gripper = 0.0
    for action in actions:
        action_pose += action[:6]
        action_gripper = action[-1]  # Use the last gripper value
    return np.concatenate([action_pose, [action_gripper]], axis=axis)


def ros_msg_to_observation(env, msg) -> np.ndarray:  # noqa: ANN001
    """Convert a ROS message to an observation."""
    if "camera" in msg.channel.topic:
        return env.cameras[0].ros_msg_to_image(msg.ros_msg)
    if "pose" in msg.channel.topic:
        obs = (
            Pose.from_ros_msg(msg.ros_msg)
            .to_array(env.config.orientation_representation)
            .astype(np.float32)
        )
        return obs
    if "wrench" in msg.channel.topic:
        return np.array(
            [
                msg.ros_msg.wrench.force.x,
                msg.ros_msg.wrench.force.y,
                msg.ros_msg.wrench.force.z,
                msg.ros_msg.wrench.torque.x,
                msg.ros_msg.wrench.torque.y,
                msg.ros_msg.wrench.torque.z,
            ],
            dtype=np.float32,
        )
    if "joint_states" in msg.channel.topic and "gripper" not in msg.channel.topic:
        return env.robot.ros_msg_to_joint(msg.ros_msg).astype(np.float32)
    if "action" in msg.channel.topic:
        return np.array(msg.ros_msg.data, dtype=np.float32)
    if "gripper" in msg.channel.topic:
        return np.array([env.gripper.ros_msg_to_gripper_value(msg.ros_msg)], dtype=np.float32)
    raise ValueError(f"Unknown topic: {msg.channel.topic}")


def get_message_count(mcap_file: Path | str) -> int:
    """Get the total number of messages in an MCAP file.

    Args:
        mcap_file (str): Path to the MCAP file.

    Returns:
        int: Total number of messages in the MCAP file.
    """
    mcap_file = Path(mcap_file)
    with open(mcap_file, "rb") as f:
        reader = NonSeekingReader(f)
        summary = reader.get_summary()
        if summary is None:
            raise ValueError("No summary found in MCAP file")
    return summary.statistics.message_count if summary.statistics else 0


def get_fps_from_recording(mcap_file: Path | str) -> int:
    """Estimate the FPS from the MCAP recording based on the timestamps of the messages.

    Args:
        mcap_file (str): Path to the MCAP file.

    Returns:
        int | None: Estimated FPS, or None if it cannot be determined.
    """
    timestamps = []
    for msg in read_ros2_messages(source=mcap_file, topics=["/action"]):
        timestamps.append(msg.log_time_ns)
    if len(timestamps) < 2:
        raise ValueError("Not enough messages to estimate FPS from /action topic.")

    deltas = np.diff(timestamps) / 1e9  # Convert to seconds
    mean_delta = np.mean(deltas)
    fps = round(1 / mean_delta)

    return fps


def convert_mcap_file_to_lerobot_episode(
    env: ManipulatorBaseEnv,
    dataset: LeRobotDataset,
    mcap_file: Path | str,
):
    """Convert an MCAP file to LeRobot Dataset format and upload to Hugging Face Hub.

    Args:
        env (ManipulatorBaseEnv): Environment to use for defining features.
        mcap_file (str): Path to the MCAP file.
        dataset (LeRobotDataset): LeRobotDataset instance to add frames to.
    """
    mcap_file = Path(mcap_file)
    assert mcap_file.exists(), f"MCAP file {mcap_file} does not exist."

    topics_to_features = env.get_topics_to_features()
    topics_to_features["/action"] = "action"

    filtered_features = {}
    for _, feature in topics_to_features.items():
        if feature in dataset.features.keys():
            filtered_features[feature] = dataset.features[feature].copy()

    latest_observation: dict[str, np.ndarray | None] = {
        feature: None for feature in filtered_features.keys()
    }

    message_count = get_message_count(mcap_file)

    actual_fps = get_fps_from_recording(mcap_file)

    if actual_fps != dataset.fps:
        print(
            f"[yellow]Warning:[/yellow] The FPS of the dataset ({dataset.fps}) does not match the estimated FPS from the recording ({actual_fps})."
        )
        print("Actions will be combined to match the dataset FPS.")

    fps_ratio = actual_fps / dataset.fps
    if fps_ratio < 1.0:
        raise ValueError(
            f"The dataset FPS ({dataset.fps}) cannot be higher than the recording FPS ({actual_fps})."
        )
    action_buffer = []

    for msg in track(
        read_ros2_messages(source=mcap_file, topics=[*topics_to_features.keys()]),
        total=message_count,
        description=f"Processing MCAP file {mcap_file} - repo {dataset.repo_id} - fps {dataset.fps}",
    ):
        latest_observation[topics_to_features[msg.channel.topic]] = ros_msg_to_observation(env, msg)
        if "action" in msg.channel.topic:
            if actual_fps != dataset.fps:
                action_buffer.append(latest_observation["action"])
                if len(action_buffer) >= fps_ratio:
                    combined_action = combine_actions(action_buffer, axis=0)
                    latest_observation["action"] = combined_action
                    action_buffer = []
                else:
                    continue
            if any([observation is None for observation in latest_observation.values()]):
                continue
            dataset.add_frame(latest_observation, task="random")

    dataset.save_episode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert an MCAP file to LeRobot Dataset format and upload to Hugging Face Hub."
    )
    parser.add_argument(
        "--mcap_file",
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
        required=True,
        help="Name of the environment to use.",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="",
        help="Namespace for the environment (if applicable).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="FPS of the dataset. If not provided, it will be estimated from the /action topic in the MCAP file.",
    )
    args = parser.parse_args()

    mcap_file = args.mcap_file
    repo_id = args.repo_id
    env_name = args.env_name
    namespace = args.namespace
    fps = args.fps or get_fps_from_recording(mcap_file)

    if fps is None:
        fps = get_fps_from_recording(mcap_file)
        print(f"Estimated FPS from recording: {fps}")

    env = make_env(env_name, namespace=namespace)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        features=get_features(env),
        fps=fps,
        use_videos=True,
    )

    try:
        convert_mcap_file_to_lerobot_episode(env, dataset, mcap_file)
    except Exception as e:
        env.close()
        raise e

    env.close()
