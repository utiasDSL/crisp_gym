"""Asynchronous Lerobot Policy Module."""

import logging
from collections import deque
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Callable, Tuple

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import get_policy_class
from lerobot.policies.utils import populate_queues

try:
    from lerobot.utils.constants import OBS_IMAGES
except ImportError:
    from lerobot.constants import OBS_IMAGES
from typing_extensions import override

from crisp_gym.envs.manipulator_env import ManipulatorBaseEnv
from crisp_gym.policy.policy import Action, Observation, Policy, register_policy
from crisp_gym.util.lerobot_features import concatenate_state_features, numpy_obs_to_torch


@register_policy("async_lerobot_policy")
class AsyncLerobotPolicy(Policy):
    """Asynchronous Lerobot Policy."""

    def __init__(self, pretrained_path: str, env: ManipulatorBaseEnv):
        """Initialize the policy."""
        self.parent_conn, self.child_conn = Pipe()
        self.env = env
        # ToDo: make these parameters not hardcoded
        self.n_obs = 2
        self.n_act = 4
        self.replan_time = 2
        self.inpainting = False

        self.inf_proc = Process(
            target=inference_worker,
            kwargs={
                "conn": self.child_conn,
                "pretrained_path": pretrained_path,
                "env": env,
                "steps": self.n_act,
                "inpainting": self.inpainting,
                "replan_time": self.replan_time,
            },
            daemon=True,
        )
        self.inf_proc.start()

    @override
    def make_data_fn(self) -> Callable[[], Tuple[Observation, Action]]:  # noqa: ANN002, ANN003
        """Return a function that returns (obs, action) each frame by talking to the worker.

        Behaviour:
         - On first call: collect n_obs observations into a rolling buffer.
         - Request chunks according to the replan_time parameter.
         - Each call executes one action from the current chunk and returns (obs, action) for sorting/recording.
        """
        # Before starting, fill the observation buffer
        obs_buf: deque = deque(maxlen=self.n_obs)
        for _ in range(self.n_obs):
            obs_buf.append(self.env._get_obs())

        # Prepare first chunk in the case that n_act != replan_time
        if self.n_act != self.replan_time:
            self.parent_conn.send({"type": "OBS_SEQ", "obs_seq": list(obs_buf)})
            print("Starting new inference")

        i = 0
        next_chunk = None
        current_chunk = None

        def _fn() -> tuple:
            nonlocal i, next_chunk, current_chunk  # Required to mutate across calls
            if i == 0:
                if (
                    self.n_act == self.replan_time
                ):  # Edge case when we want to make a new prediction after all action chunks have been used up
                    obs_buf.append(self.env._get_obs())
                    self.parent_conn.send({"type": "OBS_SEQ", "obs_seq": list(obs_buf)})
                    print("Starting new inference")
                next_chunk = self.parent_conn.recv()
                current_chunk = next_chunk[self.n_act - self.replan_time :]
                print("Length ot the new current chunk:", len(current_chunk))

            # execute action
            action = current_chunk[i]
            print("Process element:", i)
            obs, *_ = self.env.step(action, block=False)
            obs_buf.append(obs)

            # Start prediction
            if i == (2 * self.replan_time - self.n_act):
                self.parent_conn.send({"type": "OBS_SEQ", "obs_seq": list(obs_buf)})
                print("Starting new inference")

            # step done
            i += 1

            # when done with one episode reset the counter
            if i >= (len(current_chunk)):
                i = 0

            return obs, action

        return _fn

    @override
    def reset(self):
        """Reset the policy state."""
        self.parent_conn.send("reset")

    @override
    def shutdown(self):
        """Shutdown the policy and release resources."""
        self.parent_conn.send(None)
        _drain_conn(self.parent_conn)
        self.inf_proc.join()


def inference_worker(  # noqa: D417
    conn: Connection,
    pretrained_path: str,
    env: ManipulatorBaseEnv,
    steps: int | None,
    inpainting: bool,
    replan_time: int,
):  # noqa: ANN001
    """Policy inference process: loads policy on GPU, receives observations via conn, returns actions, and exits on None.

    Args:
        conn (Connection): The connection to the parent process for sending and receiving data.
        pretrained_path (str): Path to the pretrained policy model.
        env (ManipulatorBaseEnv): The environment in which the policy will be applied.
        steps (int): How many actions are executed from the prediction
        inpainting (bool): Wether to use inpainting in the prediction of a new chunk or not
        replan_time (int): After how many steps to start predicting a new action chunk
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_config = TrainPipelineConfig.from_pretrained(pretrained_path)
    if train_config.policy is None:
        raise ValueError(
            f"Policy configuration is missing in the pretrained path: {pretrained_path}. "
            "Please ensure the policy is correctly configured."
        )
    policy_cls = get_policy_class(train_config.policy.type)

    policy_config = PreTrainedConfig.from_pretrained(pretrained_path)

    if steps is not None:
        # Check if the number of steps make sense
        horizon = policy_config.horizon
        if steps >= horizon:
            raise ValueError(
                f"The policy steps={steps} must be smaller than the horizon={horizon}."
                "Please modify your cli."
            )
        policy_config.n_action_steps = int(steps)

    if inpainting is True:
        policy_config.inpainting_lengh = max(
            0, int(policy_config.n_action_steps) - int(replan_time)
        )

    policy = policy_cls.from_pretrained(pretrained_path, config=policy_config)

    logging.info(
        f"[Inference] Loaded {policy.name} policy with {pretrained_path} on device {device}."
    )

    policy.reset()
    policy.to(device).eval()

    # Read policy config to know obs/action window sizes
    cfg = policy.config
    n_obs = int(cfg.n_obs_steps)
    print("Ready to recive information")

    while True:
        # Check if messages are recieved correctly
        msg = conn.recv()
        if msg is None:
            break
        if msg == "reset":
            logging.info("[Inference] Resetting policy")
            policy.reset()
            continue
        if not (isinstance(msg, dict) and msg.get("type") == "OBS_SEQ"):
            logging.warning(f"[Inference] Unknown message: {type(msg)}")
            continue

        # We are recieving a list of dictonaries with the last observations
        obs_seq = msg["obs_seq"]

        # Make the policy predict an action chunk for the current obeservation.
        # Therefore we follow the implementation on the Lerobot side for select_action() which calls predict_action_chunk()
        with torch.inference_mode():
            for i in range(n_obs):
                last = obs_seq[i]

                last["observation.state"] = concatenate_state_features(last)
                batch = numpy_obs_to_torch(last)

                # This mirrors Lerobot `select_action()` pre-processing so queues are filled correctly
                batch_norm = policy.normalize_inputs(batch)
                if policy.config.image_features:
                    batch_norm = dict(batch_norm)  # shallow copy then add OBS_IMAGES stack
                    batch_norm[OBS_IMAGES] = torch.stack(
                        [batch_norm[k] for k in policy.config.image_features], dim=-4
                    )
                # Note: It's important that this happens after stacking the images into a single key.
                policy._queues = populate_queues(policy._queues, batch_norm)

            # Now get a fresh chunk
            chunk = policy.predict_action_chunk(batch_norm)
            chunk = chunk.squeeze(0).to(device="cpu").numpy()

        logging.debug(f"[Inference] Computed chunk with shape {tuple(chunk.shape)}")
        conn.send(chunk)

    conn.close()
    logging.info("[Inference] Worker shutting down")


# To be implemented later to avoid stale messages in the pipe
def _drain_conn(conn):  # noqa: ANN001
    """Non-blocking: remove any pending messages so we don't reuse stale chunks."""
    try:
        while conn.poll(0):
            _ = conn.recv()
    except (EOFError, OSError):
        pass
