import collections
import copy

import torch
from torch.nn import functional as F
from stable_baselines3.sac import SAC
from typing import Optional, Union, Dict, Type
import numpy as np
import torch as th
from multimexmf.models.pretrain_models import EnsembleMLP, Normalizer
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.utils import polyak_update
from multimexmf.commons.intrinsic_reward_algorithms.utils import BaseIntrinsicReward


class SACEipo(SAC):
    def __init__(self,
                 ensemble_model_kwargs: Dict,
                 intrinsic_reward_weights: Optional[Dict] = None,
                 agg_intrinsic_reward: str = 'mean',
                 normalize_ensemble_training: bool = True,
                 pred_diff: bool = True,
                 intrinsic_reward_model: Optional[Type[BaseIntrinsicReward]] = None,
                 extrinsic_reward_weight: Union[float, str] = 'auto',
                 max_intrinsic_reward_lambda: float = 10 ** 8,
                 normalize_intrinsic_reward: bool = True,
                 *args,
                 **kwargs
                 ):
        self.normalize_ensemble_training = normalize_ensemble_training
        self.pred_diff = pred_diff
        self.exploitation_ent_coef_optimizer: Optional[th.optim.Adam] = None
        self.exploitation_log_ent_coef = None
        self.extrinsic_reward_weight = extrinsic_reward_weight
        self.max_intrinsic_reward_lambda = max_intrinsic_reward_lambda
        self.normalize_intrinsic_reward = normalize_intrinsic_reward
        super().__init__(*args, **kwargs)
        self._setup_ensemble_model(
            ensemble_model_kwargs=ensemble_model_kwargs,
            intrinsic_reward_weights=intrinsic_reward_weights,
            intrinsic_reward_model=intrinsic_reward_model,
            device=self.device,
            agg_intrinsic_reward=agg_intrinsic_reward,
        )

    def _setup_model(self):
        super()._setup_model()
        if self.extrinsic_reward_weight == 'auto':
            self.log_extrinsic_reward_weight = th.log(th.ones(1, device=self.device)).requires_grad_(True)
            self.max_log_intrinsic_reward_lambda = th.log(th.ones(1, device=self.device)
                                                          * self.max_intrinsic_reward_lambda)
            self.extrinsic_reward_weight_optimizer = th.optim.Adam([self.log_extrinsic_reward_weight],
                                                                   lr=self.lr_schedule(1))
            self.extrinsic_reward_weight = None
        else:
            self.extrinsic_reward_weight = th.tensor([self.extrinsic_reward_weight], device=self.device).squeeze()
            self.extrinsic_reward_weight_optimizer = None
            self.log_extrinsic_reward_weight = None

        self.exploitation_policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,
        )
        self.exploitation_policy.to(self.device)
        self.exploration_policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,
        )
        self.exploration_policy.to(self.device)
        self._create_exploitation_aliases()

        self.exploitation_batch_norm_stats = copy.deepcopy(self.batch_norm_stats)
        self.exploitation_batch_norm_stats_target = copy.deepcopy(self.batch_norm_stats_target)

        self.exploration_batch_norm_stats = copy.deepcopy(self.batch_norm_stats)
        self.exploration_batch_norm_stats_target = copy.deepcopy(self.batch_norm_stats_target)

    def _create_exploitation_aliases(self):
        self.exploitation_actor = self.exploitation_policy.actor
        self.exploitation_critic = self.exploitation_policy.critic
        self.exploitation_critic_target = self.exploitation_policy.critic_target
        self.exploration_critic = self.exploration_policy.critic
        self.exploration_critic_target = self.exploration_policy.critic_target

    def _setup_normalizer(self, input_dim: int, output_dict: Dict, device: th.device):
        self.input_normalizer = Normalizer(input_dim=input_dim, update=self.normalize_ensemble_training,
                                           device=device)
        output_normalizers = {}
        for key, val in output_dict.items():
            output_normalizers[key] = Normalizer(input_dim=val.shape[-1], update=self.normalize_ensemble_training,
                                                 device=device)
        self.output_normalizers = output_normalizers
        self.int_reward_normalizer = Normalizer(input_dim=1, update=self.normalize_intrinsic_reward,
                                                device=device)

    def _setup_ensemble_model(self,
                              ensemble_model_kwargs: Dict,
                              intrinsic_reward_weights: Dict,
                              intrinsic_reward_model: Union[Type[BaseIntrinsicReward], None],
                              device: th.device,
                              agg_intrinsic_reward: str = 'sum',
                              ) -> None:
        sample_obs = self.observation_space.sample()
        if isinstance(sample_obs, Dict):
            for key in sample_obs.keys():
                sample_obs[key] = np.expand_dims(sample_obs[key], 0)
        else:
            sample_obs = np.expand_dims(sample_obs, 0)
        dummy_feat = self.extract_features(
            obs_as_tensor(sample_obs,
                          self.device)
        )
        input_dim = dummy_feat.shape[-1] + self.action_space.shape[0]
        output_dict = self._get_ensemble_targets(sample_obs,
                                                 sample_obs,
                                                 rewards=torch.zeros((1, 1))
                                                 )
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

    def extract_features(self, obs):
        with th.no_grad():
            features = self.actor.extract_features(
                obs, features_extractor=self.actor.features_extractor)
        return features

    def _get_ensemble_targets(self, next_obs: Union[th.Tensor, Dict], obs: Union[th.Tensor, Dict],
                              rewards: th.Tensor) -> Dict:
        if self.pred_diff:
            assert type(next_obs) == type(obs)
            if isinstance(next_obs, np.ndarray) or isinstance(next_obs, dict):
                next_obs = obs_as_tensor(next_obs, self.device)
                obs = obs_as_tensor(obs, self.device)
            next_obs = self.extract_features(next_obs)
            obs = self.extract_features(obs)
            return {
                'next_obs': next_obs - obs,
                'reward': rewards,
            }
        else:
            if isinstance(next_obs, np.ndarray) or isinstance(next_obs, dict):
                next_obs = obs_as_tensor(next_obs, self.device)
            next_obs = self.extract_features(next_obs)
            return {
                'next_obs': next_obs,
                'reward': rewards,
            }

    def get_intrinsic_reward(self, inp: th.Tensor, labels: Dict) -> th.Tensor:
        # calculate intrinsic reward
        if self.intrinsic_reward_model is None:
            return th.zeros(inp.shape[0], device=self.device)
        else:
            return self.intrinsic_reward_model(inp=inp, labels=labels)

    def _update_ensemble_normalizers(self, batch_size: int):
        if self.replay_buffer.size() >= batch_size:
            replay_data = self.replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)
            features = self.extract_features(replay_data.observations)
            inp = th.cat([features, replay_data.actions], dim=-1)
            self.input_normalizer.update(inp)

            labels = self._get_ensemble_targets(replay_data.next_observations, replay_data.observations,
                                                replay_data.rewards)
            for key, y in labels.items():
                self.output_normalizers[key].update(y)

    def train_exploitation_critic(self, replay_data):
        """Train policy and critic for maximizing only the extrinsic reward.
            Policy is not used for exploration in the real environment -> entropy term is not required for this policy.
        """
        if self.use_sde:
            self.exploitation_actor.noise()

        actions_pi, log_prob = self.exploitation_actor.action_log_prob(replay_data.observations)

        # ent_coef_loss = None
        # if self.exploitation_ent_coef_optimizer is not None and self.exploitation_log_ent_coef is not None:
        #     # Important: detach the variable from the graph
        #     # so we don't change it with other losses
        #     # see https://github.com/rail-berkeley/softlearning/issues/60
        #     ent_coef = th.exp(self.exploitation_log_ent_coef.detach())
        #     ent_coef_loss = -(self.exploitation_log_ent_coef * (log_prob +
        #                                                         self.exploitation_target_entropy).detach()).mean()
        # else:
        #     ent_coef = self.exploitation_ent_coef_tensor

        # if ent_coef_loss is not None and self.exploitation_ent_coef_optimizer is not None:
        #     self.exploitation_ent_coef_optimizer.zero_grad()
        #     ent_coef_loss.backward()
        #     self.exploitation_ent_coef_optimizer.step()

        with th.no_grad():
            # Select action according to policy
            next_actions, next_log_prob = self.exploitation_actor.action_log_prob(replay_data.next_observations)
            # Compute the next Q values: min over all critics targets
            next_q_values = th.cat(self.exploitation_critic_target(replay_data.next_observations, next_actions), dim=1)
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            # td error + entropy term
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        current_q_values = self.exploitation_critic(replay_data.observations, replay_data.actions)

        # Compute critic loss
        critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
        assert isinstance(critic_loss, th.Tensor)  # for type checker

        # Optimize the critic
        self.exploitation_critic.optimizer.zero_grad()
        critic_loss.backward()
        self.exploitation_critic.optimizer.step()

        # Compute actor loss
        # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
        # Min over all critic networks
        q_values_pi = th.cat(self.exploitation_critic(replay_data.observations, actions_pi), dim=1)
        min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
        actor_loss = -min_qf_pi.mean()

        # Optimize the actor
        self.exploitation_actor.optimizer.zero_grad()
        actor_loss.backward()
        self.exploitation_actor.optimizer.step()
        return critic_loss, actor_loss

    def train_explore_exploit_policy(self, replay_data):
        """Train policy to maximize intrinsic + extrinsic reward"""
        if self.use_sde:
            self.actor.noise()
        actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
        log_prob = log_prob.reshape(-1, 1)

        ent_coef_loss = None
        if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            ent_coef = th.exp(self.log_ent_coef.detach())
            ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
        else:
            ent_coef = self.ent_coef_tensor

        if self.log_extrinsic_reward_weight is not None:
            extrinsic_reward_weight = th.exp(self.log_extrinsic_reward_weight.detach())
        else:
            extrinsic_reward_weight = self.extrinsic_reward_weight

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()

        with th.no_grad():
            # Select action according to policy
            next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
            # Compute the next Q values: min over all critics targets
            next_q_values = th.cat(self.critic_target(
                replay_data.next_observations, next_actions), dim=1)
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            # Note: We only add entropy term for the exploration critic
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # td error + entropy term
            # relabel reward with exploration reward
            labels = self._get_ensemble_targets(replay_data.next_observations, replay_data.observations,
                                                replay_data.rewards)
            features = self.extract_features(replay_data.observations)
            inp = th.cat([features, replay_data.actions], dim=-1)
            # normalize inputs when gradient step == normalization_index which is randomly sampled.
            # if gradient_step == normalization_index:
            #    self.input_normalizer.update(inp)
            inp = self.input_normalizer.normalize(inp)
            for key, y in labels.items():
                # if gradient_step == normalization_index:
                #    self.output_normalizers[key].update(y)
                labels[key] = self.output_normalizers[key].normalize(y)
            rewards = self.get_intrinsic_reward(
                inp=inp,
                labels=labels
            ).reshape(-1, 1)
            rewards = self.int_reward_normalizer.normalize(rewards)
            # termination flag is not used for exploration critic
            next_q_exploration_values = th.cat(self.exploration_critic_target(
                replay_data.next_observations, next_actions), dim=1)
            next_q_exploration_values, _ = th.min(next_q_exploration_values, dim=1, keepdim=True)
            # add entropy term
            next_q_exploration_values = next_q_exploration_values - ent_coef * next_log_prob.reshape(-1, 1)
            # no termination flag used for the exploration policy
            target_exploration_q_values = rewards + self.gamma * next_q_exploration_values

        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        current_q_values = self.critic(replay_data.observations, replay_data.actions)
        current_exploration_q_values = self.exploration_critic(replay_data.observations, replay_data.actions)
        # Compute critic loss
        critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
        assert isinstance(critic_loss, th.Tensor)  # for type checker
        exploration_critic_loss = 0.5 * sum(F.mse_loss(current_q, target_exploration_q_values)
                                            for current_q in current_exploration_q_values)
        assert isinstance(exploration_critic_loss, th.Tensor)

        # Optimize the critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        self.exploration_critic.optimizer.zero_grad()
        exploration_critic_loss.backward()
        self.exploration_critic.optimizer.step()

        # Compute actor loss
        # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
        # Min over all critic networks
        q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
        min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
        exploration_q_values = th.cat(self.exploration_critic(replay_data.observations, actions_pi), dim=1)
        min_qf_explore_pi, _ = th.min(exploration_q_values, dim=1, keepdim=True)
        min_qf_explore_pi = min_qf_explore_pi - ent_coef * log_prob
        value = min_qf_pi + (min_qf_explore_pi / extrinsic_reward_weight)
        actor_loss = -value.mean()

        # Optimize the actor
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
        return ent_coef, ent_coef_loss, extrinsic_reward_weight, rewards, \
            inp, labels, critic_loss, exploration_critic_loss, actor_loss

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        self.exploitation_policy.set_training_mode(True)
        self.exploration_policy.set_training_mode(True)
        self.ensemble_model.train(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer,
                      self.exploitation_actor.optimizer, self.exploitation_critic.optimizer,
                      self.exploration_critic.optimizer,
                      ]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        if self.extrinsic_reward_weight_optimizer is not None:
            optimizers += [self.extrinsic_reward_weight_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        extrinsic_reward_weight_losses, extrinsic_reward_weights = [], []
        actor_losses, critic_losses, exploration_critic_losses = [], [], []

        exploitation_actor_losses, exploitation_critic_losses = [], []
        batch_intrinsic_reward = []

        ensemble_losses = collections.defaultdict(list)

        self._update_ensemble_normalizers(batch_size)

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            exploit_critic_loss, exploit_actor_loss = self.train_exploitation_critic(replay_data)
            exploitation_actor_losses.append(exploit_actor_loss.item())
            exploitation_critic_losses.append(exploit_critic_loss.item())

            ent_coef, ent_coef_loss, extrinsic_reward_weight, int_rewards, inp, labels, \
                critic_loss, exploration_critic_loss, actor_loss = self.train_explore_exploit_policy(replay_data)

            self.int_reward_normalizer.update(int_rewards)

            ent_coefs.append(ent_coef.item())
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            exploration_critic_losses.append(exploration_critic_loss.item())
            batch_intrinsic_reward.append(int_rewards.mean().item())
            if ent_coef_loss is not None:
                ent_coef_losses.append(ent_coef_loss.item())
            extrinsic_reward_weights.append(extrinsic_reward_weight.item())

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.exploration_critic.parameters(),
                              self.exploration_critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
                polyak_update(self.exploration_batch_norm_stats, self.exploration_batch_norm_stats_target, 1.0)

                # repeat for exploration critic
                polyak_update(self.exploitation_critic.parameters(), self.exploitation_critic_target.parameters(),
                              self.tau)
                polyak_update(self.exploitation_batch_norm_stats, self.exploitation_batch_norm_stats_target, 1.0)

            # Update extrinsic reward weight
            if self.extrinsic_reward_weight_optimizer is not None:
                # get actions from policies
                actions, _ = self.actor.action_log_prob(replay_data.next_observations)
                expl_actions, _ = self.exploitation_actor.action_log_prob(replay_data.observations)

                # take value for V^{\pi_{E+I}}_{E}: exploration + exploitation policy under
                # its critic of extrinsic rewards
                critic_value_actor = th.cat(self.critic(replay_data.observations, actions), dim=1)
                critic_value_actor, _ = th.min(critic_value_actor, dim=1, keepdim=True)

                # take value for V^{\pi_E}_{E}: exploitation policy under its critic of extrinsic reward
                exploitation_actor_value = th.cat(self.exploitation_critic(replay_data.observations,
                                                                                  expl_actions), dim=1)
                exploitation_actor_value, _ = th.min(exploitation_actor_value, dim=1, keepdim=True)

                # ensure that V^{\pi_{E+I}}_{E} = V^{\pi_E}_{E}
                extrinsic_reward_weight_loss = torch.min(self.log_extrinsic_reward_weight,
                                                         self.max_log_intrinsic_reward_lambda) * \
                                  ((critic_value_actor - exploitation_actor_value).mean().detach())
                self.extrinsic_reward_weight_optimizer.zero_grad()
                extrinsic_reward_weight_loss.backward()
                self.extrinsic_reward_weight_optimizer.step()
                extrinsic_reward_weight_losses.append(extrinsic_reward_weight_loss.item())

            # ensemble model training
            self.ensemble_model.optimizer.zero_grad()
            prediction = self.ensemble_model(inp)
            loss = self.ensemble_model.loss(prediction=prediction, target=labels)
            stacked_losses = []
            for key, val in loss.items():
                ensemble_losses[key].append(val.item())
                stacked_losses.append(val)
            stacked_losses = th.stack(stacked_losses)
            total_loss = stacked_losses.mean()
            total_loss.backward()
            self.ensemble_model.optimizer.step()

        self._n_updates += gradient_steps
        self.ensemble_model.train(False)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/exploitation_actor_losses", np.mean(exploitation_actor_losses))
        self.logger.record("train/exploitation_critic_losses", np.mean(exploitation_critic_losses))
        for key, val in ensemble_losses.items():
            self.logger.record(f"train/ensemble_loss_{key}", np.mean(val))
            self.logger.record(f"train/out_normalizer_mean_{key}", np.mean(
                self.output_normalizers[key].mean.cpu().numpy()))
            self.logger.record(f"train/out_normalizer_std_{key}", np.mean(
                self.output_normalizers[key].std.cpu().numpy()))
        self.logger.record(f"train/inp_normalizer_mean", np.mean(self.input_normalizer.mean.cpu().numpy()))
        self.logger.record(f"train/inp_normalizer_std", np.mean(self.input_normalizer.std.cpu().numpy()))
        self.logger.record(f"train/int_reward_normalizer_mean", np.mean(self.int_reward_normalizer.mean.cpu().numpy()))
        self.logger.record(f"train/int_reward_normalizer_std", np.mean(self.int_reward_normalizer.std.cpu().numpy()))
        self.logger.record("train/batch_intrinsic_reward", np.mean(batch_intrinsic_reward))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        if len(extrinsic_reward_weight_losses) > 0:
            self.logger.record("train/extrinsic_reward_weight_loss", np.mean(extrinsic_reward_weight_losses))
        self.logger.record("train/extrinsic_reward_weight", np.mean(extrinsic_reward_weights))

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            tb_log_name: str = "SACEipo",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ):
        super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar
        )


if __name__ == '__main__':
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback
    from gymnasium.envs.classic_control.pendulum import PendulumEnv
    from gymnasium.wrappers.time_limit import TimeLimit
    from typing import Optional
    from multimexmf.commons.intrinsic_reward_algorithms.utils import \
        DisagreementIntrinsicReward
    from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder

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
            u = np.clip(u, -self.max_torque, self.max_torque)[0]
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


    log_dir = '../intrinsic_reward_algorithms/check/'
    n_envs = 4
    vec_env = make_vec_env(CustomPendulumEnv, n_envs=1, seed=0, wrapper_class=TimeLimit,
                           env_kwargs={'render_mode': 'rgb_array'},
                           wrapper_kwargs={'max_episode_steps': 200})

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
        VecVideoRecorder(
            make_vec_env(CustomPendulumEnv, n_envs=4, seed=1,
                         env_kwargs={'render_mode': 'rgb_array'},
                         wrapper_class=TimeLimit,
                         wrapper_kwargs={'max_episode_steps': 200}
                         ),
            video_folder=log_dir + 'eval/',
            record_video_trigger=lambda x: True,
        ),
        log_path=log_dir,
        best_model_save_path=log_dir,
        eval_freq=1000,
        n_eval_episodes=5,
        deterministic=True,
    )
    algorithm_kwargs = {
        'policy': 'MlpPolicy',
        # 'train_freq': 32,
        # 'gradient_steps': 32,
        'learning_rate': 1e-3,
        'verbose': 1,
        'gradient_steps': -1,
    }

    ensemble_model_kwargs = {
        'learn_std': False,
        'optimizer_kwargs': {'lr': 3e-4, 'weight_decay': 0.0},
    }
    maximize_entropy = True
    if maximize_entropy:
        algorithm = SACEipo(
            env=vec_env,
            extrinsic_reward_weight="auto",
            intrinsic_reward_model=DisagreementIntrinsicReward,
            ensemble_model_kwargs=ensemble_model_kwargs,
            **algorithm_kwargs
        )
    else:
        algorithm = SAC(
            env=vec_env,
            **algorithm_kwargs,
        )
    algorithm.learn(
        total_timesteps=100_000,
        # callback=eval_callback,
    )
