from stable_baselines3.common.buffers import RolloutBuffer, DictRolloutBuffer, DictReplayBuffer, \
    DictReplayBufferSamples
from stable_baselines3.common.type_aliases import TensorDict, ReplayBufferSamples
from gymnasium import spaces
from typing import Union, Optional, NamedTuple
import torch as th
import numpy as np
from typing import Any, Dict, List, Optional, Union

from stable_baselines3.common.preprocessing import get_obs_shape
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecEnv, VecNormalize


class IntrinsicRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_intrinsic_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    intrinsic_advantages: th.Tensor
    returns: th.Tensor
    intrinsic_returns: th.Tensor


class DictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class IntrinsicRewardRolloutBuffer(RolloutBuffer):
    non_episodic_intrinsic_reward: bool = True,

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            intrinsic_model_buffer_size: int = 1_000_000,
            device: Union[th.device, str] = "auto",
            gae_lambda: float = 1,
            gamma: float = 0.99,
            n_envs: int = 1,
            intrinsic_gae_lambda: Optional[float] = None,
            intrinsic_gamma: Optional[float] = None,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs
        )
        if intrinsic_gae_lambda:
            self.intrinsic_gae_lambda = intrinsic_gae_lambda
        else:
            self.intrinsic_gae_lambda = gae_lambda
        if intrinsic_gamma:
            self.intrinsic_gamma = intrinsic_gamma
        else:
            self.intrinsic_gamma = gamma

        # setup buffer for intrinsic reward
        self._intrinsic_model_buffer_size = intrinsic_model_buffer_size
        self._intrinsic_model_inp_dim = None
        self._intrinsic_model_labels = None

    def reset(self) -> None:
        super().reset()
        self.intrinsic_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.intrinsic_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.intrinsic_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.intrinsic_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def setup_intrinsic_model_buffer(self, intrinsic_model_inp_dim: int, intrinsic_model_labels: Dict):
        self._intrinsic_model_inp_dim = intrinsic_model_inp_dim
        self._intrinsic_model_labels = intrinsic_model_labels
        self._reset_intrinsic_model_data()

    def _reset_intrinsic_model_data(self):
        assert isinstance(self._intrinsic_model_inp_dim, int) or isinstance(self._intrinsic_model_labels, Dict)
        self._intrinsic_model_inp = np.zeros((self._intrinsic_model_buffer_size, self.n_envs,
                                              self._intrinsic_model_inp_dim))
        self._intrinsic_model_out = {}
        for key, label_input_shape in self._intrinsic_model_labels.items():
            self._intrinsic_model_out[key] = np.zeros((self._intrinsic_model_buffer_size, self.n_envs,
                                                       *label_input_shape), dtype=np.float32)
        self._intrinsic_model_pos = 0
        self._intrinsic_model_full = False

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """

        assert last_values.shape[-1] == 2, "need to return both intrinsic and extrinsic values"

        # Convert to numpy
        last_values = last_values.clone().cpu().numpy()
        last_values, intrinsic_last_values = last_values[..., 0].flatten(), last_values[..., -1].flatten()

        last_gae_lam = 0
        last_intr_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
                next_intrinsic_values = intrinsic_last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
                next_intrinsic_values = self.intrinsic_values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            if self.non_episodic_intrinsic_reward:
                intrinsic_delta = self.intrinsic_rewards[step] + self.intrinsic_gamma * next_intrinsic_values \
                                  - self.intrinsic_values[step]
                last_intr_gae_lam = intrinsic_delta + \
                                    self.intrinsic_gamma * self.intrinsic_gae_lambda * last_intr_gae_lam
            else:
                intrinsic_delta = self.intrinsic_rewards[step] + \
                                  self.intrinsic_gamma * next_intrinsic_values * next_non_terminal \
                                  - self.intrinsic_values[step]
                last_intr_gae_lam = intrinsic_delta + \
                                    self.intrinsic_gamma * self.intrinsic_gae_lambda * \
                                    next_non_terminal * last_intr_gae_lam

            self.advantages[step] = last_gae_lam
            self.intrinsic_advantages[step] = last_intr_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values
        self.intrinsic_returns = self.intrinsic_advantages + self.intrinsic_values

    def add(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            episode_start: np.ndarray,
            value: th.Tensor,
            log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        assert reward.shape[-1] == 2 and value.shape[-1] == 2
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)
        reward, intrinsic_reward = reward[..., 0], reward[..., -1]
        value, intrinsic_value = value[..., 0], value[..., -1]
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.intrinsic_rewards[self.pos] = np.array(intrinsic_reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy()
        self.intrinsic_values[self.pos] = intrinsic_value.clone().cpu().numpy()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def add_intrinsic_model_data(self, inp, labels):
        for key in self._intrinsic_model_labels.keys():
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            self._intrinsic_model_out[key][self._intrinsic_model_pos] = labels[key].clone().cpu().numpy()

        self._intrinsic_model_inp[self._intrinsic_model_pos] = inp.clone().cpu().numpy()
        self._intrinsic_model_pos += 1
        if self._intrinsic_model_pos == self._intrinsic_model_buffer_size:
            self._intrinsic_model_full = True
            self._intrinsic_model_pos = 0

    def get(self, batch_size: Optional[int] = None):
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "intrinsic_values",
                "log_probs",
                "advantages",
                "intrinsic_advantages",
                "returns",
                "intrinsic_returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
            self,
            batch_inds: np.ndarray,
            env=None,
    ) -> IntrinsicRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.intrinsic_values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.intrinsic_advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.intrinsic_returns[batch_inds].flatten(),
        )
        return IntrinsicRolloutBufferSamples(*tuple(map(self.to_torch, data)))

    def intrinsic_model_samples(self, batch_size: int):
        upper_bound = self._intrinsic_model_buffer_size if self._intrinsic_model_full else self._intrinsic_model_pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        inp = self.to_torch(self._intrinsic_model_inp[batch_inds, env_indices, :]).float()
        labels = {key: self.to_torch(label[batch_inds, env_indices, :]).float() for key, label in
                  self._intrinsic_model_out.items()}
        return inp, labels


class NStepDictReplayBuffer(DictReplayBuffer):
    #     """
    #     Peng's Q(lambda) replay buffer
    #     Paper: https://arxiv.org/abs/2103.00107
    #
    #     .. warning::
    #
    #       For performance reasons, the maximum number of steps per episodes must be specified.
    #       In most cases, it will be inferred if you specify ``max_episode_steps`` when registering the environment
    #       or if you use a ``gym.wrappers.TimeLimit`` (and ``env.spec`` is not None).
    #       Otherwise, you can directly pass ``max_episode_length`` to the replay buffer constructor.
    #
    #     :param buffer_size: The size of the buffer measured in transitions.
    #     :param max_episode_length: The maximum length of an episode. If not specified,
    #         it will be automatically inferred if the environment uses a ``gym.wrappers.TimeLimit`` wrapper.
    #     :param device: PyTorch device
    #     :param handle_timeout_termination: Handle timeout termination (due to timelimit)
    #         separately and treat the task as infinite horizon task.
    #         https://github.com/DLR-RM/stable-baselines3/issues/284
    #     """
    #
    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Dict,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
            gamma: float = 0.99,
    ):
        self.gamma = gamma
        super().__init__(buffer_size=buffer_size,
                         observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         n_envs=n_envs,
                         optimize_memory_usage=optimize_memory_usage,
                         handle_timeout_termination=handle_timeout_termination)

    def set_gamma(self, gamma: float):
        self.gamma = gamma

    def _get_samples(  # type: ignore[override]
            self,
            batch_inds: np.ndarray,
            env: Optional[VecNormalize] = None,
            n_steps: Optional[int] = 1,
    ) -> DictReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()},
                                   env)
        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}

        next_obs_ = {key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()}

        assert isinstance(obs_, dict)
        assert isinstance(next_obs_, dict)
        reward = self.rewards[batch_inds, env_indices]
        done = self.dones[batch_inds, env_indices]
        timeout = self.timeouts[batch_inds, env_indices]
        gamma = np.ones_like(done)
        for idx in range(1, n_steps):
            reward = reward * done + (1 - done) * (reward + self.gamma
                                                   * gamma * self.rewards[batch_inds + idx, env_indices])
            timeout = timeout * done + (1 - done) * self.timeouts[batch_inds + idx, env_indices]
            gamma = gamma * (done + (1 - done) * self.gamma)
            next_obs_ = \
                {key: done.reshape((-1, )
                                   + (1,) * len(next_obs_[key].shape[1:])) * next_obs_[key]
                      + (1 - done.reshape((-1, )
                                   + (1,) * len(next_obs_[key].shape[1:]))) * obs[batch_inds + idx, env_indices, :]
                 for key, obs in self.next_observations.items()}
            done = done * done + (1 - done) * self.dones[batch_inds + idx, env_indices]

        next_obs_ = self._normalize_obs(next_obs_, env)
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        # termination flag
        # Only use dones that are not due to timeouts
        # deactivated by default (timeouts is initialized as an array of False)
        dones = self.to_torch(
                done * (1 - timeout)).reshape(
                -1, 1
            )
        # discount factor correction
        not_dones = (1 - dones) * gamma.reshape(-1, 1)
        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations=next_observations,
            dones=1 - not_dones,
            rewards=self.to_torch(self._normalize_reward(reward.reshape(-1, 1), env)),
        )

    def sample(self,
               batch_size: int,
               env: Optional[VecNormalize] = None,
               n_steps: int = 1,
               ):
        upper_bound = self.buffer_size if self.full else self.pos
        if upper_bound - n_steps > 0:
            upper_bound = upper_bound - n_steps
        else:
            n_steps = 1
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env, n_steps=n_steps)

#
#         self.n_steps = n_steps
#         self.gamma = gamma
#         super(NStepDictReplayBuffer, self).__init__(
#             buffer_size=buffer_size,
#             observation_space=observation_space,
#             action_space=action_space,
#             device=device,
#             n_envs=n_envs,
#             optimize_memory_usage=optimize_memory_usage)
#         # Handle timeouts termination properly if needed
#         # see https://github.com/DLR-RM/stable-baselines3/issues/284
#         self.handle_timeout_termination = handle_timeout_termination
#         buffer_size = self.max_episode_stored
#         self.observations = {
#             key: np.zeros((buffer_size, self.n_steps, self.n_envs, *_obs_shape), dtype=observation_space[key].dtype)
#             for key, _obs_shape in self.obs_shape.items()
#         }
#         self.next_observations = {
#             key: np.zeros((buffer_size, self.n_steps, self.n_envs, *_obs_shape), dtype=observation_space[key].dtype)
#             for key, _obs_shape in self.obs_shape.items()
#         }
#
#         self.actions = np.zeros(
#             (buffer_size, self.n_steps, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
#         )
#
#         self.rewards = np.zeros((buffer_size, self.n_steps, self.n_envs), dtype=np.float32)
#         self.dones = np.zeros((buffer_size, self.n_steps, self.n_envs), dtype=np.float32)
#         # Handle timeouts termination properly if needed
#         # see https://github.com/DLR-RM/stable-baselines3/issues/284
#         self.timeouts = np.zeros((buffer_size, self.n_steps, self.n_envs), dtype=np.float32)
#
#         # episode length storage, needed for episodes which has less steps than the maximum length
#         self.episode_lengths = np.zeros(self.max_episode_stored, self.n_envs, dtype=np.int64)
#         self.episode_steps = 0
#
#     def add(
#             self,
#             obs: Dict[str, np.ndarray],
#             next_obs: Dict[str, np.ndarray],
#             action: np.ndarray,
#             reward: np.ndarray,
#             done: np.ndarray,
#             infos: List[Dict[str, Any]],
#     ) -> None:
#
#         # Remove termination signals due to timeout
#         if self.handle_timeout_termination:
#             done_ = done * (1 - np.array([info.get("TimeLimit.truncated", False) for info in infos]))
#         else:
#             done_ = done
#
#         for key in self.observations.keys():
#             # Reshape needed when using multiple envs with discrete observations
#             # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
#             if isinstance(self.observation_space.spaces[key], spaces.Discrete):
#                 obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
#             self.observations[key][self.pos][self.current_idx] = np.array(obs[key])
#
#         for key in self.next_observations.keys():
#             if isinstance(self.observation_space.spaces[key], spaces.Discrete):
#                 next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
#             self.next_observations[key][self.pos][self.current_idx] = np.array(next_obs[key])
#
#         self.actions[self.pos][self.current_idx] = action
#         self.dones[self.pos][self.current_idx] = done_
#         self.rewards[self.pos][self.current_idx] = reward
#
#         # update current pointer
#         self.current_idx += 1
#         self.episode_steps += 1
#
#         if done or self.episode_steps >= self.n_steps:
#             self.store_episode()
#             self.episode_steps = 0
#
#     def store_episode(self) -> None:
#         """
#         Increment episode counter
#         and reset transition pointer.
#         """
#         # add episode length to length storage
#         self.episode_lengths[self.pos] = self.current_idx
#
#         self.pos += 1
#         if self.pos == self.max_episode_stored:
#             self.full = True
#             self.pos = 0
#         # reset transition pointer
#         self.current_idx = 0
#
#     @property
#     def n_episodes_stored(self) -> int:
#         if self.full:
#             return self.max_episode_stored
#         return self.pos
#
#     def size(self) -> int:
#         """
#         :return: The current number of transitions in the buffer.
#         """
#         return int(np.sum(self.episode_lengths))
#
#     @property
#     def max_episode_stored(self):
#         return self.buffer_size // self.n_steps
#
#     def reset(self) -> None:
#         """
#         Reset the buffer.
#         """
#         self.pos = 0
#         self.current_idx = 0
#         self.full = False
#         self.episode_lengths = np.zeros(self.max_episode_stored, dtype=np.int64)
#
#     def sample(  # type: ignore[override]
#         self,
#         batch_size: int,
#         env: Optional[VecNormalize] = None,
#     ):
#         upper_bound = self.max_episode_stored if self.full else self.pos
#         batch_inds = np.random.randint(0, upper_bound, size=batch_size)
#         return self._get_samples(batch_inds, env=env)
#
#     def _get_samples(  # type: ignore[override]
#         self,
#         batch_inds: np.ndarray,
#         env: Optional[VecNormalize] = None,
#     ):
#         # Sample randomly the env idx
#         env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
#
#         # Normalize if needed and remove extra dimension (we are using only one env for now)
#         obs_ = self._normalize_obs({key: obs[batch_inds, 0, env_indices, :] for key, obs in self.observations.items()},
#                                    env)
#         episode_lengths = self.episode_lengths[batch_inds]
#         next_obs_ = self._normalize_obs(
#             {key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()}, env
#         )
#         for i in range(self.n_steps):
#
#
#         assert isinstance(obs_, dict)
#         assert isinstance(next_obs_, dict)
#         # Convert to torch tensor
#         observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
#         next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}
#
#         return DictReplayBufferSamples(
#             observations=observations,
#             actions=self.to_torch(self.actions[batch_inds, env_indices]),
#             next_observations=next_observations,
#             # Only use dones that are not due to timeouts
#             # deactivated by default (timeouts is initialized as an array of False)
#             dones=self.to_torch(
#                 self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(
#                 -1, 1
#             ),
#             rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
#         )


if __name__ == '__main__':
    dones = np.sort(1 * (np.random.uniform(low=0, high=1, size=(50, 1)) > 0.5), axis=0)
    rewards = np.random.uniform(low=0, high=1, size=50)
    targ_gamma = 0.9
    gamma = 1
    done = dones[0]
    reward = rewards[0]
    first_done = np.where(dones == 1)[0][0]
    print(first_done, targ_gamma ** first_done, np.cumsum(rewards[:first_done + 1])[-1])
    for i in range(1, 50):
        gamma *= (done + (1 - done) * targ_gamma)
        reward = reward * done + (1-done) * (reward + rewards[i])
        done = done * done + (1 - done) * dones[i]
        print(i, done, gamma, reward)


