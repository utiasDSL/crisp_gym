"""Asynchronous Lerobot Policy Module."""

from crisp_gym.policy.policy import Policy, register_policy


@register_policy("async_lerobot_policy")
class AsyncLerobotPolicy(Policy):
    """Asynchronous Lerobot Policy."""

    raise NotImplementedError("AsyncLerobotPolicy is not yet implemented.")
