from gym import wrappers


class MBEnvWrapper:
    """
    Wrapper for gym environments.
    To be used with this model-based RL codebase (PDDM).
    """

    def __init__(self, env):

        self.env = env.env
        self.unwrapped_env = env.env.env

        self.action_dim = self.unwrapped_env.action_space.shape[0]

    def reset(self, reset_state=None, return_start_state=False):

        if reset_state:
            reset_pose = reset_state["reset_pose"]
            reset_vel = reset_state["reset_vel"]
            reset_goal = reset_state["reset_goal"]

            # reset to specified state
            obs = self.unwrapped_env.do_reset(reset_pose, reset_vel, reset_goal)

        else:

            # standard reset call
            obs = self.unwrapped_env.reset_model()

            # pose
            if hasattr(self.unwrapped_env, "reset_pose"):
                reset_pose = self.unwrapped_env.reset_pose.copy()
            else:
                reset_pose = None

            # vel
            if hasattr(self.unwrapped_env, "reset_vel"):
                reset_vel = self.unwrapped_env.reset_vel.copy()
            else:
                reset_vel = None

            # goal
            if hasattr(self.unwrapped_env, "reset_goal"):
                reset_goal = self.unwrapped_env.reset_goal.copy()
            else:
                reset_goal = None

        # save relevant state info needed to reset in future
        reset_state = dict(
            reset_pose=reset_pose, reset_vel=reset_vel, reset_goal=reset_goal
        )

        # return
        if return_start_state:
            return obs, reset_state
        else:
            return obs

    def step(self, action):
        return self.unwrapped_env.step(action)
