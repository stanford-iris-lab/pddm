from gym.envs.registration import register

register(
    id="pddm_baoding-v0",
    entry_point="pddm.envs.baoding.baoding_env:BaodingEnv",
    max_episode_steps=1000,
)
