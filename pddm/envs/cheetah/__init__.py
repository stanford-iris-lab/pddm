from gym.envs.registration import register

register(
    id="pddm_cheetah-v0",
    entry_point="pddm.envs.cheetah.cheetah:HalfCheetahEnv",
    max_episode_steps=1000,
)
