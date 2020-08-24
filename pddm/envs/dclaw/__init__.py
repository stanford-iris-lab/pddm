from gym.envs.registration import register

register(
    id="pddm_dclaw_turn-v0",
    entry_point="pddm.envs.dclaw.dclaw_turn_env:DClawTurnEnv",
    max_episode_steps=500,
)
