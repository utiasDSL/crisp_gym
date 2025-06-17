import gymnasium as gym
import numpy as np
import pinocchio as pin
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation

from crisp_py.robot import Robot
from crisp_py.gripper.gripper import Gripper
from crisp_py.devices.camera import Camera
from crisp_py.envs.manipulator_env_config import ManipulatorEnvConfig, FrankaEnvConfig

class ManipulatorBaseEnv(gym.Env):
    """Base class for Manipulator Gym Environment.
    This class serves as a base for creating specific Manipulator Gym environments.
    It cannot be used directly.
    """

    def __init__(self, namespace: str = "", config: Optional[ManipulatorEnvConfig] = None):
        """Initialize the Manipulator Gym Environment.

        Args:
            config (ManipulatorEnvConfig): Configuration for the environment.
        """
        super().__init__()
        self.config = config if config else FrankaEnvConfig()

        self.robot = Robot(namespace=namespace, robot_config=self.config.robot_config)
        self.gripper = Gripper(namespace=namespace, 
                               gripper_config=self.config.gripper_config,
                               )
        self.cameras = [
            Camera(namespace=namespace, 
                   config=camera_config, 
                   ) for camera_config in self.config.camera_configs
        ]

        self.timestep = 0
        self.ctrl_type = None

        self.robot.wait_until_ready(timeout=3)
        self.gripper.wait_until_ready(timeout=3)
        for camera in self.cameras:
            camera.wait_until_ready(timeout=3)

        self.control_rate = self.robot.node.create_rate(self.config.control_frequency)

        self.observation_space = gym.spaces.Dict(
            {
                **{
                    f'{camera.config.camera_name}_image': gym.spaces.Box(
                        low=np.zeros((*camera.config.resolution, 3), dtype=np.uint8),
                        high=255 * np.ones((*camera.config.resolution, 3), dtype=np.uint8),
                        dtype=np.uint8,
                    )
                    for camera in self.cameras
                },
                "joint": gym.spaces.Box(
                    low=np.ones((7, ), dtype=np.float32) * -np.pi,
                    high=np.ones((7, ), dtype=np.float32) * np.pi,
                    dtype=np.float32
                ),
                "cartesian": gym.spaces.Box(
                    low=-np.ones((6, ), dtype=np.float32),
                    high=np.ones((6, ), dtype=np.float32),
                    dtype=np.float32
                ),
                "gripper": gym.spaces.Box(
                    low=np.zeros((1, ), dtype=np.float32),
                    high=np.ones((1, ), dtype=np.float32),
                    dtype=np.float32
                ),
            }
        )

    def _get_obs(self) -> dict:
        """Get the current observation from the robot.

        Returns:
            dict: Current observation.
        """
        obs = {}
        for camera in self.cameras:
            obs[f'{camera.config.camera_name}_image'] = camera.current_image
        obs["joint"] = self.robot.joint_values

        eef_pose = np.array(self.robot.end_effector_pose)
        obs["cartesian"] = np.concatenate((eef_pose[:3, 3], Rotation.from_matrix(eef_pose[:3, :3]).as_euler('xyz')))

        obs["gripper"] = 1 - np.array([self.gripper.value])
        return obs
    
    def _set_gripper_action(self, action: float):
        if action >= self.config.gripper_threshold and self.gripper.is_open():
            self.gripper.close()
        elif action < self.config.gripper_threshold and not self.gripper.is_open():
            self.gripper.open()

    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, bool, dict]:
        obs = {}
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        if self.config.max_episode_steps and self.timestep >= self.config.max_episode_steps:
            truncated = True

        self.timestep += 1

        return obs, reward, terminated, truncated, info
    
    def reset(self) -> Tuple[dict, dict]:
        """Reset the environment."""
        super().reset()

        self.timestep = 0

        return self._get_obs(), {}

    def close(self):
        """Close the environment."""
        self.robot.shutdown()
        super().close()

    def switch_controller(self, ctrl_type: str):
        ctrl_types = {'joint': 'joint_impedance_controller', 'cartesian': 'cartesian_impedance_controller'}

        if ctrl_type not in ctrl_types:
            print(f'Controller {ctrl_type} not availabe.')

            return 
        
        self.ctrl_type = ctrl_type
        self.robot.controller_switcher_client.switch_controller(ctrl_types[self.ctrl_type])

    def home(self, home_config: list[float] | None = None, blocking: bool = True):
        current_ctrl_type = self.ctrl_type

        self.gripper.open()
        self.robot.home(home_config, blocking)
        

        self.switch_controller(current_ctrl_type)

    def move_to(self, position: iter = None, pose: iter = None, speed: float = 0.05):

        current_ctrl_type = self.ctrl_type

        self.switch_controller('cartesian')
        
        if pose:
            pose = pin.SE3(quat=pin.Quaternion(Rotation.from_euler('xyz', np.array(pose)).as_quat()), translation=np.array(position))
            position = None

        self.gripper.open()
        self.robot.move_to(position=position, pose=pose, speed=speed)

        self.robot.reset_targets()
        self.robot.wait_until_ready()
        self.switch_controller(current_ctrl_type)


class ManipulatorCartesianEnv(ManipulatorBaseEnv):
    """Manipulator Cartesian Environment.

    This class is a specific implementation of the Manipulator Gym Environment for Cartesian space control.
    """
    def __init__(self, namespace: str = "", config: Optional[ManipulatorEnvConfig] = None):
        """Initialize the Manipulator Cartesian Environment.

        Args:
            config (ManipulatorEnvConfig): Configuration for the environment.
        """
        super().__init__(namespace, config)

        self._min_z_height = 0.0

        self.action_space = gym.spaces.Box(
            low=np.concatenate([
                -np.ones((3, ), dtype=np.float32),
                -np.ones((3, ), dtype=np.float32) * np.pi,
                np.zeros((1, ), dtype=np.float32)
            ], axis=0),
            high=np.concatenate([
                np.ones((3, ), dtype=np.float32),
                np.ones((3, ), dtype=np.float32) * np.pi,
                np.ones((1, ), dtype=np.float32)
            ], axis=0),
            dtype=np.float32
        )

        self.switch_controller("cartesian")

    def step(self, action: np.ndarray, block: bool = True) -> Tuple[dict, float, bool, bool, dict]:

        assert action.shape == self.action_space.shape, f"Action shape {action.shape} does not match expected shape {self.action_space.shape}"
        #assert self.action_space.contains(action), f"Action {action} is not in the action space {self.action_space}"

        target_pose = self.robot.target_pose

        quat = pin.Quaternion(target_pose.rotation)
        current_position, current_orientation = target_pose.translation, Rotation.from_quat([quat.x, quat.y, quat.z, quat.w] )
        translation, rotation = action[:3], Rotation.from_euler('xyz', action[3:6])

        target_position = current_position + translation
        target_position[2] = max(target_position[2], self._min_z_height)
        target_orientation = rotation * current_orientation

        target_pose = pin.SE3(quat=pin.Quaternion(target_orientation.as_quat()), translation=target_position)
        self.robot.set_target(pose=target_pose)

        self._set_gripper_action(action[6])

        if block:
            self.control_rate.sleep()

        _, reward, terminated, truncated, info = super().step(action)

        obs = self._get_obs()

        return obs, reward, terminated, truncated, info

class ManipulatorJointEnv(ManipulatorBaseEnv):
    """Manipulator Joint Environment.

    This class is a specific implementation of the Manipulator Gym Environment for Joint space control.
    """

    def __init__(self, namespace: str = "", config: Optional[ManipulatorEnvConfig] = None):
        """Initialize the Manipulator Joint Environment.

        Args:
            config (ManipulatorEnvConfig): Configuration for the environment.
        """
        super().__init__(namespace, config)

        self.action_space = gym.spaces.Box(
            low=np.concatenate([
                np.ones((7, ), dtype=np.float32) * -np.pi,
                np.zeros((1, ), dtype=np.float32)
            ], axis=0),
            high=np.concatenate([
                np.ones((7, ), dtype=np.float32) * np.pi,
                np.ones((1, ), dtype=np.float32)
            ], axis=0),
            dtype=np.float32
        )

        self.switch_controller("joint")

    def step(self, action: np.ndarray, block: bool = True) -> Tuple[dict, float, bool, bool, dict]:
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        assert action.shape == self.action_space.shape, f"Action shape {action.shape} does not match expected shape {self.action_space.shape}"
        #assert self.action_space.contains(action), f"Action {action} is not in the action space {self.action_space}"

        target_joint = (self.robot.target_joint + action[:7] + np.pi) % (2 * np.pi) - np.pi

        self.robot.set_target_joint(target_joint)

        self._set_gripper_action(action[7])

        if block:
            self.control_rate.sleep()

        _, reward, terminated, truncated, info = super().step(action)

        obs = self._get_obs()

        return obs, reward, terminated, truncated, info

