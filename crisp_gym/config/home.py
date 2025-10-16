"""Contains some home configurations."""
# TODO: make the configs robot specific

from enum import Enum

home_close_to_table = [
    -1.73960110e-02,
    9.55319758e-02,
    8.09703053e-04,
    -1.94272034e00,
    -4.01435784e-03,
    2.06584183e00,
    7.97426445e-01,
]

home_front_up = [
    -0.02312892,
    -0.10664185,
    -0.0195703,
    -1.75644521,
    -0.00732298,
    1.68992915,
    0.8040582,
]


class HomeConfig(Enum):
    """Enum for different home configurations."""

    CLOSE_TO_TABLE = home_close_to_table
    FRONT_UP = home_front_up

    def randomize(self, noise: float = 0.01) -> list:
        """Randomize the home configuration."""
        import numpy as np

        return (
            np.array(self.value) + np.random.uniform(-noise, noise, size=len(self.value))
        ).tolist()
