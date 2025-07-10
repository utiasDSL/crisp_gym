"""Environment wrappers for extending Gym environments with additional functionality.

This module provides wrappers that add features like observation stacking and receding
horizon control to Gym environments. These wrappers can be used to modify the behavior
of environments without changing their core implementation.

The module includes:
    - WindowWrapper: Stacks a fixed-size window of past observations along a new time dimension
    - RecedingHorizon: Applies a sequence of actions in a receding horizon manner
    - stack_gym_space: Helper function to repeat/stack Gym spaces
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import rclpy
from numpy.typing import NDArray

from crisp_gym.manipulator_env import ManipulatorBaseEnv


def stack_gym_space(space: gym.Space, repeat: int) -> gym.Space:
    """Repeat a Gym space definition by stacking it multiple times.

    Args:
        space (gym.Space): The original Gym space to be repeated.
        repeat (int): Number of times to repeat/stack the space.

    Returns:
        gym.Space: A new Gym space with the original space stacked 'repeat' times.

    Raises:
        ValueError: If the input space type is not supported (currently supports Box and Dict spaces).
    """
    if isinstance(space, gym.spaces.Box):
        # Convert dtype to type to match Box constructor's type annotation
        dtype = np.dtype(space.dtype).type
        return gym.spaces.Box(
            low=np.repeat(space.low[None], repeat, axis=0),
            high=np.repeat(space.high[None], repeat, axis=0),
            dtype=dtype,
        )
    elif isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict({k: stack_gym_space(v, repeat) for k, v in space.spaces.items()})
    else:
        raise ValueError(f"Space {space} is not supported.")


class WindowWrapper(gym.Wrapper):
    """A Gym wrapper that stacks a fixed-size window of past observations along a new time dimension.

    This allows agents to receive a temporal context of the environment by maintaining a history
    of the most recent `window_size` observations. The wrapper modifies the observation space
    to include the temporal dimension, making it compatible with policies that expect
    temporal information.

    Attributes:
        window_size (int): Number of consecutive observations to stack.
        window (list): List of most recent observations.
        observation_space (gym.Space): Modified observation space that includes temporal dimension.
    """

    def __init__(self, env: ManipulatorBaseEnv, window_size: int) -> None:
        """Initialize the WindowWrapper.

        Args:
            env (ManipulatorBaseEnv): The environment to wrap.
            window_size (int): Number of consecutive observations to stack.
        """
        super().__init__(env)
        self.window_size = window_size
        self.window = []
        self.observation_space = stack_gym_space(self.env.observation_space, self.window_size)

    def step(
        self, action: NDArray[np.float32], **kwargs: Any
    ) -> Tuple[Dict[str, NDArray[np.float32]], float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment.

        Args:
            action (np.ndarray): An action provided by the agent.
            **kwargs: Additional keyword arguments passed to the environment's step function.

        Returns:
            tuple:
                - dict: The current observation.
                - float: Amount of reward returned after previous action.
                - bool: Whether the episode has ended.
                - bool: Whether the episode was truncated.
                - dict: Contains auxiliary diagnostic information.
        """
        obs, reward, terminated, truncated, info = self.env.step(action, **kwargs)
        self.window.append(obs)
        self.window = self.window[-self.window_size :]
        obs = {
            key: np.stack([frame[key] for frame in self.window]) for key in self.window[0].keys()
        }
        return obs, float(reward), terminated, truncated, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, NDArray[np.float32]], Dict[str, Any]]:
        """Reset the environment to an initial state.

        Args:
            seed (int, optional): The seed that is used to initialize the environment's PRNG.
            options (dict, optional): Additional information to specify how the environment is reset.

        Returns:
            tuple:
                - dict: The initial observation.
                - dict: Additional information.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        self.window = [obs] * self.window_size
        obs = {
            key: np.stack([frame[key] for frame in self.window]) for key in self.window[0].keys()
        }
        return obs, info

    def close(self) -> None:
        """Clean up the environment's resources."""
        self.env.close()

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the wrapped environment.

        Args:
            name (str): Name of the attribute to get.

        Returns:
            Any: The value of the requested attribute.
        """
        return getattr(self.env, name)


class RecedingHorizon(gym.Wrapper):
    """A Gym wrapper that takes a sequence of actions and applies them in a receding horizon manner.

    This wrapper allows the agent to plan and execute a sequence of actions over a fixed time horizon.
    At each step, the agent provides a sequence of actions, and the wrapper executes them sequentially
    until either the horizon is reached or the episode terminates.
    """

    def __init__(self, env: ManipulatorBaseEnv, horizon_length: int) -> None:
        """Initialize the RecedingHorizon wrapper.

        Args:
            env (ManipulatorBaseEnv): The environment to wrap.
            horizon_length (int): The number of steps to look ahead and execute actions for.
        """
        super().__init__(env)
        self.horizon_length = horizon_length
        self.action_space = stack_gym_space(self.env.action_space, self.horizon_length)

    def step(
        self, action: NDArray[np.float32], **kwargs: Any
    ) -> Tuple[Dict[str, NDArray[np.float32]], float, bool, bool, Dict[str, Any]]:
        """Execute a sequence of actions over the horizon length.

        Args:
            action (np.ndarray): A sequence of actions to execute. Shape should be
                (horizon_length, action_dim) or (action_dim,) for horizon_length=1.
            **kwargs: Additional keyword arguments passed to the environment's step function.

        Returns:
            tuple:
                - dict: The final observation after executing all actions.
                - float: Sum of rewards received over the horizon.
                - bool: Whether the episode has ended.
                - bool: Whether the episode was truncated.
                - dict: Contains auxiliary diagnostic information.

        Raises:
            AssertionError: If the action sequence length is less than horizon_length.
        """
        obs = {}
        rewards = []
        terminated = False
        truncated = False
        info = {}

        if self.horizon_length == 1 and len(action.shape) == 1:
            action = action[None]
        assert action.shape[0] >= self.horizon_length

        for i in range(self.horizon_length):
            obs, reward, terminated, truncated, info = self.env.step(action[i], **kwargs)
            rewards.append(reward)
            if terminated or truncated:
                break

        return obs, float(np.sum(rewards)), terminated, truncated, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, NDArray[np.float32]], Dict[str, Any]]:
        """Reset the environment to an initial state.

        Args:
            seed (int, optional): The seed that is used to initialize the environment's PRNG.
            options (dict, optional): Additional information to specify how the environment is reset.

        Returns:
            tuple:
                - dict: The initial observation.
                - dict: Additional information.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def close(self) -> None:
        """Clean up the environment's resources."""
        if rclpy.ok():
            rclpy.shutdown()
        self.env.close()

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the wrapped environment.

        Args:
            name (str): Name of the attribute to get.

        Returns:
            Any: The value of the requested attribute.
        """
        return getattr(self.env, name)
