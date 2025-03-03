# import the environment wrapper and gymnasium
from skrl.envs.wrappers.torch import wrap_env
import gymnasium as gym

# load a vectorized environment
env = gym.make_vec("Pendulum-v1", num_envs=10, vectorization_mode="async")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gymnasium")'