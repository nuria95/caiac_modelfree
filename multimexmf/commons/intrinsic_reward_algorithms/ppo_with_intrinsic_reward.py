from stable_baselines3.ppo import PPO
from typing import Optional, Union, Dict, Type, Any
from gymnasium import spaces
import numpy as np
import torch as th
from multimexmf.models.pretrain_models import EnsembleMLP, Normalizer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.buffers import RolloutBuffer, ReplayBuffer, DictReplayBuffer
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import get_schedule_fn
from multimexmf.commons.intrinsic_reward_algorithms.utils import BaseIntrinsicReward
from copy import deepcopy


class PPOWithIntrinsicReward(PPO):
    def __init__(self,
                 ensemble_model_kwargs: Dict,
                 exploration_weight_schedule: Union[float, Schedule],
                 intrinsic_reward_weights: Optional[Dict] = None,
                 agg_intrinsic_reward: str = 'sum',
                 intrinsic_reward_gradient_steps: int = 1000,
                 intrinsic_reward_batch_size: int = 64,
                 normalize_ensemble_training: bool = True,
                 model_replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
                 model_replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
                 model_buffer_size: int = 1_000_000,
                 pred_diff: bool = True,
                 intrinsic_reward_model: Optional[Type[BaseIntrinsicReward]] = None,
                 *args,
                 **kwargs
                 ):
        self.normalize_ensemble_training = normalize_ensemble_training
        self.intrinsic_reward_batch_size = intrinsic_reward_batch_size
        self.intrinsic_reward_gradient_steps = intrinsic_reward_gradient_steps
        self.exploration_weight_schedule = exploration_weight_schedule
        self.model_replay_buffer_class = model_replay_buffer_class
        self.model_replay_buffer_kwargs = model_replay_buffer_kwargs or {}
        self.model_replay_buffer = None
        self.model_buffer_size = model_buffer_size
        self.pred_diff = pred_diff

        super().__init__(*args, **kwargs)
        self._setup_ensemble_model(
            ensemble_model_kwargs=ensemble_model_kwargs,
            intrinsic_reward_weights=intrinsic_reward_weights,
            intrinsic_reward_model=intrinsic_reward_model,
            device=self.device,
            agg_intrinsic_reward=agg_intrinsic_reward,
        )

    def _setup_exploration_weight_schedule(self) -> None:
        """Transform to callable if needed."""
        self.exploration_weight_schedule = get_schedule_fn(self.exploration_weight_schedule)

    def _setup_normalizer(self, input_dim: int, output_dict: Dict, device: th.device):
        self.input_normalizer = Normalizer(input_dim=input_dim, update=self.normalize_ensemble_training,
                                           device=device)
        output_normalizers = {}
        for key, val in output_dict.items():
            output_normalizers[key] = Normalizer(input_dim=val.shape[-1], update=self.normalize_ensemble_training,
                                                 device=device)
        self.output_normalizers = output_normalizers

    def _setup_ensemble_model(self,
                              ensemble_model_kwargs: Dict,
                              intrinsic_reward_weights: Dict,
                              intrinsic_reward_model: Union[Type[BaseIntrinsicReward], None],
                              device: th.device,
                              agg_intrinsic_reward: str = 'sum',
                              ) -> None:
        self._setup_exploration_weight_schedule()
        dummy_feat = self.extract_features(
            obs_as_tensor(self.observation_space.sample().reshape(1, -1),
                          self.device)
        )
        input_dim = dummy_feat.shape[-1] + self.action_space.shape[0]
        output_dict = self._get_ensemble_targets(self.observation_space.sample(), self.observation_space.sample())
        self.ensemble_model = EnsembleMLP(
            input_dim=input_dim,
            output_dict=output_dict,
            **ensemble_model_kwargs,
        )
        self.ensemble_model.to(device)
        self._setup_normalizer(input_dim=input_dim, output_dict=output_dict, device=device)

        if intrinsic_reward_weights is not None:
            assert intrinsic_reward_weights.keys() == output_dict.keys()
        else:
            intrinsic_reward_weights = {k: 1.0 for k in output_dict.keys()}

        if intrinsic_reward_model:
            self.intrinsic_reward_model = intrinsic_reward_model(
                intrinsic_reward_weights=intrinsic_reward_weights,
                ensemble_model=self.ensemble_model,
                agg_intrinsic_reward=agg_intrinsic_reward,
            )
        else:
            self.intrinsic_reward_model = None

        if self.model_replay_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.model_replay_buffer_class = DictReplayBuffer
            else:
                self.model_replay_buffer_class = ReplayBuffer

        if self.model_replay_buffer is None:
            # Make a local copy as we should not pickle
            # the environment when using HerReplayBuffer
            model_replay_buffer_kwargs = self.model_replay_buffer_kwargs.copy()

            self.model_replay_buffer = self.model_replay_buffer_class(
                self.model_buffer_size,
                self.observation_space,
                self.action_space,
                device=self.device,
                n_envs=self.n_envs,
                **model_replay_buffer_kwargs,
            )

    def _get_ensemble_targets(self, next_obs: Union[th.Tensor, Dict], obs: Union[th.Tensor, Dict]) -> Dict:
        if self.pred_diff:
            assert type(next_obs) == type(obs)
            if isinstance(next_obs, np.ndarray) or isinstance(next_obs, dict):
                return {
                    'next_obs': obs_as_tensor(next_obs - obs, self.device),
                }
            elif isinstance(next_obs, th.Tensor):
                return {
                    'next_obs': next_obs - obs,
                }
            else:
                raise NotImplementedError
        else:
            if isinstance(next_obs, np.ndarray) or isinstance(next_obs, dict):
                return {
                    'next_obs': obs_as_tensor(next_obs, self.device),
                }
            elif isinstance(next_obs, th.Tensor):
                return {
                    'next_obs': next_obs,
                }
            else:
                raise NotImplementedError

    def _update_exploration_weight(self) -> None:
        self.exploration_weight = self.exploration_weight_schedule(self._current_progress_remaining)
        assert 0 <= self.exploration_weight <= 1, "weight must be between 0 and 1"

    def get_intrinsic_reward(self, inp: th.Tensor, labels: Dict) -> th.Tensor:
        # normalize inputs and outputs
        inp = self.input_normalizer.normalize(inp)
        for key, y in labels.items():
            labels[key] = self.output_normalizers[key].normalize(y)
        # calculate intrinsic reward
        if self.intrinsic_reward_model is None:
            return th.zeros(inp.shape[0])
        else:
            return self.intrinsic_reward_model(inp=inp, labels=labels)

    def extract_features(self, obs):
        with th.no_grad():
            features = self.policy.extract_features(
                obs, features_extractor=self.policy.features_extractor)
        return features

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer,
            n_rollout_steps: int,
    ) -> bool:
        """
                Collect experiences using the current policy and fill a ``RolloutBuffer``.
                The term rollout here refers to the model-free notion and should not
                be used with the concept of rollout used in model-based RL or planning.

                :param env: The training environment
                :param callback: Callback that will be called at each step
                    (and at the beginning and end of the rollout)
                :param rollout_buffer: Buffer to fill with rollouts
                :param n_rollout_steps: Number of experiences to collect per environment
                :return: True if function returned with at least `n_rollout_steps`
                    collected, False if callback terminated rollout prematurely.
                """
        reward_infos = {
            'true_rewards': [],
            'intrinsic_rewards': [],
        }
        assert self._last_obs is not None, "No previous observation was provided"
        self._update_exploration_weight()
        self.logger.record("rollout/exploration_weight", self.exploration_weight)
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            with th.no_grad():
                labels = self._get_ensemble_targets(new_obs, self._last_obs)
                act = th.as_tensor(actions, device=self.device)
                features = self.extract_features(obs_tensor)
                inp = th.cat([features, act], dim=-1)
                intrinsic_rewards = self.get_intrinsic_reward(inp=inp, labels=labels)

            intrinsic_rewards = intrinsic_rewards.cpu().numpy().reshape(-1)

            self.num_timesteps += env.num_envs

            reward_infos['true_rewards'].append(rewards)
            reward_infos['intrinsic_rewards'].append(intrinsic_rewards)
            if self.intrinsic_reward_model:
                rewards = rewards * (1 - self.exploration_weight) + intrinsic_rewards * self.exploration_weight

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            next_obs = deepcopy(new_obs)

            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                ):
                    if isinstance(next_obs, dict):
                        next_obs_ = infos[idx]["terminal_observation"]
                        for key in next_obs.keys():
                            next_obs[key][idx] = next_obs_[key]
                    else:
                        next_obs[idx] = infos[idx]["terminal_observation"]

                    if infos[idx].get("TimeLimit.truncated", False):
                        terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                        with th.no_grad():
                            terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                        rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self.model_replay_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                next_obs,  # type: ignore[arg-type]
                clipped_actions,  # passed the actions actually used in the mdp
                rewards,
                dones,
                infos,
            )

            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()
        self.logger.record('rollout/true_rewards', safe_mean([val
                                                              for val in reward_infos['true_rewards']]))
        self.logger.record('rollout/intrinsic_rewards', safe_mean([val
                                                              for val in reward_infos['intrinsic_rewards']]))
        return True

    def train(self) -> None:
        super().train()
        losses = []
        for step in range(self.intrinsic_reward_gradient_steps):
            # need to pass vec normalize env to get normalized obs and next obs -->
            # these are the ones used to calculate intrinsic reward during rollout
            replay_data = self.model_replay_buffer.sample(
                self.intrinsic_reward_batch_size
            )
            target = self._get_ensemble_targets(next_obs=replay_data.next_observations,
                                                obs=replay_data.observations)
            # observations should be unnormalized for t
            features = self.extract_features(replay_data.observations)
            inp = th.cat([features, replay_data.actions], dim=-1)
            self.input_normalizer.update(inp)
            inp = self.input_normalizer.normalize(inp)
            for key, y in target.items():
                self.output_normalizers[key].update(y)
                target[key] = self.output_normalizers[key].normalize(y)
            self.ensemble_model.optimizer.zero_grad()
            prediction = self.ensemble_model(inp)
            loss = self.ensemble_model.loss(prediction=prediction, target=target)
            stacked_losses = th.stack([val for val in loss.values()])
            total_loss = stacked_losses.mean()
            total_loss.backward()
            losses.append(stacked_losses)
            self.ensemble_model.optimizer.step()

        self.ensemble_model.eval()
        losses = th.stack(losses).cpu().detach().numpy().mean(axis=0)
        for index, key in enumerate(self.ensemble_model.output_dict.keys()):
            self.logger.record("train/ensemble/" + key, losses[index])



if __name__ == '__main__':
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback
    from gymnasium.envs.classic_control.pendulum import PendulumEnv
    from gymnasium.wrappers.time_limit import TimeLimit
    from typing import Optional
    from multimexmf.commons.intrinsic_reward_algorithms.utils import \
        DisagreementIntrinsicReward, sigmoid_schedule


    class CustomPendulumEnv(PendulumEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._step = 0
            self.max_step = 200
            self.action_cost = 0.5

        def step(self, u):
            obs, reward, terminate, truncate, info = super().step(u)
            self._step += 1
            if self._step >= self.max_step:
                terminate = True
            reward = reward - self.action_cost * (u ** 2)
            return obs, reward, terminate, truncate, info

        def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
            super().reset(seed=seed)
            default_state = np.array([np.pi, 0.0])
            high = np.ones_like(default_state) * 0.1
            self.state = self.np_random.uniform(low=-high, high=high) + default_state
            self.last_u = None
            self._step = 0
            if self.render_mode == "human":
                self.render()
            return self._get_obs(), {}


    log_dir = './logs2/'
    n_envs = 4
    vec_env = make_vec_env(CustomPendulumEnv, n_envs=4, seed=0, wrapper_class=TimeLimit,
                           env_kwargs={'render_mode': 'rgb_array'},
                           wrapper_kwargs={'max_episode_steps': 200})
    # vec_env = VecVideoRecorder(venv=vec_env, video_folder=log_dir,
    #                            record_video_trigger=lambda x: True,
    #                            )

    n_steps = 1024
    # eval_callback = EvalCallback(VecVideoRecorder(make_vec_env(CustomPendulumEnv, n_envs=4, seed=1,
    #                                                            env_kwargs={'render_mode': 'rgb_array'},
    #                                                            wrapper_class=TimeLimit,
    #                                                            wrapper_kwargs={'max_episode_steps': 200}
    #                                                            ),
    #                                               video_folder=log_dir + 'eval/',
    #                                               record_video_trigger=lambda x: True,
    #                                               ),
    #                              log_path=log_dir,
    #                              best_model_save_path=log_dir,
    #                              eval_freq=n_steps,
    #                              n_eval_episodes=5, deterministic=True,
    #                              render=True)
    eval_callback = EvalCallback(
        make_vec_env(CustomPendulumEnv, n_envs=4, seed=1,
                     env_kwargs={'render_mode': 'rgb_array'},
                     wrapper_class=TimeLimit,
                     wrapper_kwargs={'max_episode_steps': 200}
                     ),
        log_path=log_dir,
        best_model_save_path=log_dir,
        eval_freq=n_steps,
        n_eval_episodes=5, deterministic=True,
    )
    algorithm_kwargs = {
        'policy': 'MlpPolicy',
        'verbose': 1,
        'n_steps': n_steps,
        'gae_lambda': 0.95,
        'gamma': 0.9,
        'n_epochs': 10,
        'learning_rate': 1e-3,
        'clip_range': 0.2,
        'ent_coef': 0.0,
        'use_sde': True,
        'sde_sample_freq': 4,
        'tensorboard_log': './logs/'
    }

    ensemble_model_kwargs = {
        'learn_std': False,
        'optimizer_kwargs': {'lr': 3e-4, 'weight_decay': 1e-4}
    }

    algorithm = PPOWithIntrinsicReward(
        env=vec_env,
        ensemble_model_kwargs=ensemble_model_kwargs,
        exploration_weight_schedule=sigmoid_schedule(),
        intrinsic_reward_model=DisagreementIntrinsicReward,
        **algorithm_kwargs
    )
    algorithm.learn(
        total_timesteps=200_000,
        # callback=eval_callback,
    )
