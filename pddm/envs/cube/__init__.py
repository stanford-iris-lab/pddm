from gym.envs.registration import register

register(
    id="pddm_cube-v0",
    entry_point="pddm.envs.cube.cube_env:CubeEnv",
    max_episode_steps=1000,
)
