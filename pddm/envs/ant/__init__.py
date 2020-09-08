from gym.envs.registration import register

register(
    id="pddm_ant-v0", entry_point="pddm.envs.ant.ant:AntEnv", max_episode_steps=1000,
)
