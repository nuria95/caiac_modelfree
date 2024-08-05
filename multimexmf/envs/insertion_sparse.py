from tactile_envs.envs.insertion import InsertionEnv


class InsertionSparse(InsertionEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, u):
        obs, reward, done, truncate, info = super().step(u)
        if done:
            reward = 1000
        else:
            reward = 0
        return obs, reward, done, truncate, info
