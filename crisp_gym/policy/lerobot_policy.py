"""Interface for a Policy interacting in CRISP."""

import logging
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Callable, Tuple

import numpy as np
import torch
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import get_policy_class
from typing_extensions import override

from crisp_gym.manipulator_env import ManipulatorBaseEnv
from crisp_gym.policy.policy import Action, Observation, Policy, register_policy
from crisp_gym.util.lerobot_features import concatenate_state_features, numpy_obs_to_torch
from crisp_gym.util.setup_logger import setup_logging

logger = logging.getLogger(__name__)


@register_policy("lerobot_policy")
class LerobotPolicy(Policy):
    """Abstract base class for a Policy."""

    def __init__(self, pretrained_path: str, env: ManipulatorBaseEnv):
        """Initialize the policy."""
        self.parent_conn, self.child_conn = Pipe()
        self.env = env

        self.inf_proc = Process(
            target=inference_worker,
            kwargs={
                "conn": self.child_conn,
                "pretrained_path": pretrained_path,
                "env": env,
            },
            daemon=True,
        )
        self.inf_proc.start()

    @override
    def make_data_fn(self) -> Callable[[], Tuple[Observation, Action]]:  # noqa: ANN002, ANN003
        """Generate observation and action by communicating with the inference worker."""

        def _fn() -> tuple:
            """Function to apply the policy in the environment.

            This function observes the current state of the environment, sends the observation
            to the inference worker, receives the action, and steps the environment.

            Returns:
                tuple: A tuple containing the observation from the environment and the action taken.
            """
            logger.debug("Requesting action from policy...")
            obs_raw: Observation = self.env.get_obs()

            obs_raw["observation.state"] = concatenate_state_features(obs_raw)

            self.parent_conn.send(obs_raw)
            action: Action = self.parent_conn.recv().squeeze(0).to("cpu").numpy()
            logger.debug(f"Action: {action}")

            try:
                self.env.step(action, block=False)
            except Exception as e:
                logger.exception(f"Error during environment step: {e}")

            return obs_raw, action

        return _fn

    @override
    def reset(self):
        """Reset the policy state."""
        self.parent_conn.send("reset")

    @override
    def shutdown(self):
        """Shutdown the policy and release resources."""
        self.parent_conn.send(None)
        self.inf_proc.join()


def inference_worker(
    conn: Connection,
    pretrained_path: str,
    env: ManipulatorBaseEnv,
):  # noqa: ANN001
    """Policy inference process: loads policy on GPU, receives observations via conn, returns actions, and exits on None.

    Args:
        conn (Connection): The connection to the parent process for sending and receiving data.
        pretrained_path (str): Path to the pretrained policy model.
        dataset_metadata (LeRobotDatasetMetadata): Metadata for the dataset, if needed.
        env (ManipulatorBaseEnv): The environment in which the policy will be applied.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("[Inference] Starting inference worker...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[Inference] Using device: {device}")

        logger.info(f"[Inference] Loading training config from {pretrained_path}...")
        train_config = TrainPipelineConfig.from_pretrained(pretrained_path)
        logger.info("[Inference] Loaded training config.")

        logger.info(f"[Inference] Train config: {train_config}")

        if train_config.policy is None:
            raise ValueError(
                f"Policy configuration is missing in the pretrained path: {pretrained_path}. "
                "Please ensure the policy is correctly configured."
            )

        logger.info("[Inference] Loading policy...")
        policy_cls = get_policy_class(train_config.policy.type)
        policy = policy_cls.from_pretrained(pretrained_path)
        policy.config.do_multi_step_sampling = False
        policy.config.n_inference_steps = 1

        logger.info(
            f"[Inference] Loaded {policy.name} policy with {pretrained_path} on device {device}."
        )
        policy.reset()
        policy.to(device).eval()

        warmup_obs_raw = env.observation_space.sample()
        warmup_obs_raw["observation.state"] = concatenate_state_features(warmup_obs_raw)
        warmup_obs = numpy_obs_to_torch(warmup_obs_raw)

        logger.info("[Inference] Warming up policy...")
        elapsed_list = []
        with torch.inference_mode():
            import time

            for _ in range(100):
                start = time.time()
                _ = policy.select_action(warmup_obs)
                end = time.time()
                elapsed = end - start
                elapsed_list.append(elapsed)

            torch.cuda.synchronize()

        avg_elapsed = sum(elapsed_list) / len(elapsed_list)
        std_elapsed = np.std(elapsed_list)
        max_elapsed = max(elapsed_list)
        min_elapsed = min(elapsed_list)
        logger.info(
            f"[Inference] Warm-up timing over 100 runs: "
            f"avg={avg_elapsed * 1000:.2f}ms, std={std_elapsed * 1000:.2f}ms, max={max_elapsed * 1000:.2f}ms, min={min_elapsed * 1000:.2f}ms"
        )

        logger.info("[Inference] Warm-up complete")

        while True:
            obs_raw = conn.recv()
            if obs_raw is None:
                break
            if obs_raw == "reset":
                logger.info("[Inference] Resetting policy")
                policy.reset()
                continue

            with torch.inference_mode():
                obs = numpy_obs_to_torch(obs_raw)
                # # TODO: prepocess
                action = policy.select_action(obs)
                # # TODO: postprocess

            logger.debug(f"[Inference] Computed action: {action}")
            conn.send(action)
    except Exception as e:
        logger.exception(f"[Inference] Exception in inference worker: {e}")

    conn.close()
    logger.info("[Inference] Worker shutting down")
