import gymnasium.spaces
from gymnasium import Wrapper
from typing import Dict
import numpy as np


class ImageCompressor(Wrapper):
    def __init__(self, env):
        super(ImageCompressor, self).__init__(env)
        self.env = env
        assert isinstance(self.env.observation_space, gymnasium.spaces.Dict)
        if 'image' in self.env.observation_space.keys():
            self.img_key = 'image'
        elif 'pixels' in self.env.observation_space.keys():
            self.img_key = 'pixels'
        else:
            raise NotImplementedError

        assert isinstance(self.env.observation_space[self.img_key], gymnasium.spaces.Box)
        lb = self.env.observation_space[self.img_key].low
        assert np.min(lb) == 0
        ub = self.env.observation_space[self.img_key].high
        self.high_max = np.max(ub)
        self.scale_image = True if self.high_max < 255 else False
        self.env.observation_space[self.img_key] = gymnasium.spaces.Box(
            low=0,
            high=255,
            shape=ub.shape,
            dtype=np.uint8,
        )

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        return self.compress_obs(obs), info

    def step(
        self, action
    ):
        observation, reward, termination, truncation, info = self.env.step(action)
        return self.compress_obs(observation), reward, termination, truncation, info

    def compress_obs(self, obs):
        if self.scale_image:
            obs[self.img_key] = np.round(255 / self.high_max * obs[self.img_key])
        obs[self.img_key] = obs[self.img_key].astype(np.uint8)
        return obs
