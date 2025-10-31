from crisp_gym.manipulator_env import ManipulatorCartesianEnv, make_env_config
from crisp_gym.util.rl_utils import load_actions_safe, custom_reset

config = make_env_config("my_env")
print("Env created.")
env = ManipulatorCartesianEnv(config=config)
env.wait_until_ready()
print("Env ready.")

input("Waiting")

env.close()
