import collections

import torch
from torch.nn import functional as F
from stable_baselines3.sac import SAC
from stable_baselines3.sac.policies import get_action_dim
from typing import Optional, Union, Dict, Type
import numpy as np
import torch as th
from multimexmf.models.pretrain_models import EnsembleMLP, Normalizer, EPS
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.utils import polyak_update
from multimexmf.commons.intrinsic_reward_algorithms.utils import DisagreementIntrinsicReward
from stable_baselines3.common.type_aliases import MaybeCallback


class MaxEntropySAC(SAC):
    def __init__(self,
                 ensemble_model_kwargs: Dict,
                 ensemble_type: Type[torch.nn.Module] = EnsembleMLP,
                 intrinsic_reward_weights: Optional[Dict] = None,
                 normalize_ensemble_training: bool = True,
                 pred_diff: bool = True,
                 learn_rewards: bool = True,
                 dynamics_entropy_schedule: Optional[Schedule] = None,
                 scale_dyn_entropy_with_action_dim: bool = True,
                 target_info_tau: float = 0.25,
                 target_info_gain: Union[str, float] = 'auto',
                 *args,
                 **kwargs
                 ):
        self.normalize_ensemble_training = normalize_ensemble_training
        self.pred_diff = pred_diff
        self.learn_rewards = learn_rewards
        super().__init__(*args, **kwargs)
        self._setup_ensemble_model(
            ensemble_model_kwargs=ensemble_model_kwargs,
            intrinsic_reward_weights=intrinsic_reward_weights,
            device=self.device,
            agg_intrinsic_reward='mean',
            ensemble_type=ensemble_type,
        )
        if dynamics_entropy_schedule is None:
            self.dynamics_entropy_schedule = lambda x: float(x > 0.5)
            # return a 0 for dynamics entropy after 50 % of learning steps
        else:
            self.dynamics_entropy_schedule = dynamics_entropy_schedule

        if scale_dyn_entropy_with_action_dim:
            self.dyn_entropy_scale = get_action_dim(self.action_space)
        else:
            self.dyn_entropy_scale = 1.0

        if isinstance(target_info_gain, str):
            if target_info_gain == 'auto':
                self.target_info_gain = torch.zeros(1).squeeze()
                self.target_info_tau = target_info_tau
            else:
                raise NotImplementedError
        elif isinstance(target_info_gain, float):
            self.target_info_gain = torch.tensor([target_info_gain]).squeeze()
            self.target_info_tau = 0.0
        else:
            raise NotImplementedError

    def get_dynamics_entropy_weight(self):
        return self.dynamics_entropy_schedule(self._current_progress_remaining)

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
                              device: th.device,
                              agg_intrinsic_reward: str = 'mean',
                              ensemble_type: Type[torch.nn.Module] = EnsembleMLP,
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

        self.ensemble_model = ensemble_type(
            input_dim=input_dim,
            output_dict=output_dict,
            use_entropy=True,
            **ensemble_model_kwargs,
        )

        self.ensemble_model.to(device)
        self._setup_normalizer(input_dim=input_dim, output_dict=output_dict, device=device)

        if intrinsic_reward_weights is not None:
            assert intrinsic_reward_weights.keys() == output_dict.keys()
        else:
            intrinsic_reward_weights = {k: 1.0 for k in output_dict.keys()}

        self.intrinsic_reward_model = DisagreementIntrinsicReward(
            intrinsic_reward_weights=intrinsic_reward_weights,
            ensemble_model=self.ensemble_model,
            agg_intrinsic_reward=agg_intrinsic_reward,
        )

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
            if self.learn_rewards:
                return {
                    'next_obs': next_obs - obs,
                    'reward': rewards,
                }
            else:
                return {
                    'next_obs': next_obs - obs,
                    # 'reward': rewards,
                }
        else:
            if isinstance(next_obs, np.ndarray) or isinstance(next_obs, dict):
                next_obs = obs_as_tensor(next_obs, self.device)
            next_obs = self.extract_features(next_obs)
            if self.learn_rewards:
                return {
                    'next_obs': next_obs,
                    'reward': rewards,
                }
            else:
                return {
                    'next_obs': next_obs,
                    # 'reward': rewards,
                }

    def get_intrinsic_reward(self, inp: th.Tensor, labels: Dict) -> th.Tensor:
        # calculate intrinsic reward
        entropy = self.intrinsic_reward_model(inp=inp, labels=labels)
        if not self.ensemble_model.learn_std:
            info_gain = entropy - torch.log(torch.ones_like(entropy) * EPS)
            return info_gain * self.dyn_entropy_scale
        else:
            return entropy * self.dyn_entropy_scale

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)
        dyn_ent_weight = self.get_dynamics_entropy_weight()

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        dynamics_info_gain, target_info_gain = [], []

        ensemble_losses = collections.defaultdict(list)

        self._update_ensemble_normalizers(batch_size)

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            features = self.extract_features(replay_data.observations)
            inp = th.cat([features, actions_pi], dim=-1)
            inp = self.input_normalizer.normalize(inp)
            dynamics_entropy = self.get_intrinsic_reward(
                inp=inp,
                labels=None,
            ).reshape(-1, 1)
            batch_entropy = dynamics_entropy.mean()
            dynamics_info_gain.append(batch_entropy.item())
            batch_entropy = batch_entropy * dyn_ent_weight
            target_info_gain.append(self.target_info_gain.item())
            total_entropy = -log_prob + dynamics_entropy * dyn_ent_weight
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (-total_entropy +
                                                       self.target_info_gain + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

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
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_features = self.extract_features(replay_data.next_observations)
                # get entropy of transitions
                inp = th.cat([next_features, next_actions], dim=-1)
                inp = self.input_normalizer.normalize(inp)
                next_obs_dynamics_entropy = self.get_intrinsic_reward(
                    inp=inp,
                    labels=None,
                ).reshape(-1, 1)

                # add entropy term
                next_q_values = next_q_values - ent_coef * (-next_obs_dynamics_entropy * dyn_ent_weight
                                                            + next_log_prob.reshape(-1, 1))
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)

            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (-ent_coef * total_entropy - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
                self.target_info_gain = (1 - self.target_info_tau) * self.target_info_gain + \
                                        self.target_info_tau * batch_entropy.detach()

            # ensemble model training
            inp = th.cat([features, replay_data.actions], dim=-1)
            inp = self.input_normalizer.normalize(inp)

            labels = self._get_ensemble_targets(replay_data.next_observations, replay_data.observations,
                                                replay_data.rewards)
            for key, y in labels.items():
                # if gradient_step == normalization_index:
                #    self.output_normalizers[key].update(y)
                labels[key] = self.output_normalizers[key].normalize(y)
            self.ensemble_model.train()
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
            self.ensemble_model.eval()

        self._n_updates += gradient_steps
        self.ensemble_model.train(False)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        for key, val in ensemble_losses.items():
            self.logger.record(f"train/ensemble_loss_{key}", np.mean(val))
            self.logger.record(f"train/out_normalizer_mean_{key}", np.mean(
                self.output_normalizers[key].mean.cpu().numpy()))
            self.logger.record(f"train/out_normalizer_std_{key}", np.mean(
                self.output_normalizers[key].std.cpu().numpy()))
        self.logger.record(f"train/inp_normalizer_mean", np.mean(self.input_normalizer.mean.cpu().numpy()))
        self.logger.record(f"train/inp_normalizer_std", np.mean(self.input_normalizer.std.cpu().numpy()))
        self.logger.record("train/dynamics_info_gain", np.mean(dynamics_info_gain))
        self.logger.record("train/target_info_gain", np.mean(target_info_gain))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        self.logger.record("train/dyn_ent_weight", dyn_ent_weight)

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

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "MaxEntSac",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )


if __name__ == '__main__':
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback
    from gymnasium.envs.classic_control.pendulum import PendulumEnv
    from gymnasium.wrappers.time_limit import TimeLimit
    from typing import Optional
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
        algorithm = MaxEntropySAC(
            env=vec_env,
            ensemble_model_kwargs=ensemble_model_kwargs,
            dynamics_entropy_schedule=lambda x: float(x > 0.0),
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
