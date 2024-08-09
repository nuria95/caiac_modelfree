import collections
import copy

import torch
from torch.nn import functional as F
from stable_baselines3.sac.policies import get_action_dim
from typing import Optional, Union, Dict, Type
import numpy as np
import torch as th
from multimexmf.commons.redq import REDQ, get_probabilistic_num_min
from multimexmf.models.pretrain_models import EnsembleMLP, Normalizer, MultiHeadGaussianEnsemble, EPS
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.utils import polyak_update
from multimexmf.commons.intrinsic_reward_algorithms.utils import DisagreementIntrinsicReward
from stable_baselines3.common.type_aliases import MaybeCallback


class MaxEntropyREDQ(REDQ):
    def __init__(self,
                 ensemble_model_kwargs: Dict,
                 ensemble_type: Type[torch.nn.Module] = EnsembleMLP,
                 intrinsic_reward_weights: Optional[Dict] = None,
                 normalize_ensemble_training: bool = True,
                 pred_diff: bool = True,
                 learn_rewards: bool = True,
                 dynamics_entropy_schedule: Optional[Schedule] = None,
                 dyn_entropy_scale: float = -1,
                 model_update_delay: int = -1,
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
            ensemble_type=ensemble_type,
            device=self.device,
            agg_intrinsic_reward='mean',
        )
        if dynamics_entropy_schedule is None:
            self.dynamics_entropy_schedule = lambda x: float(x > 0.5)
            # return a 0 for dynamics entropy after 50 % of learning steps
        else:
            self.dynamics_entropy_schedule = dynamics_entropy_schedule

        if dyn_entropy_scale < 0:
            self.dyn_entropy_scale = get_action_dim(self.action_space)
        else:
            self.dyn_entropy_scale = dyn_entropy_scale
        if model_update_delay > 0:
            self.model_update_delay = model_update_delay
        else:
            self.model_update_delay = copy.deepcopy(self.policy_update_delay)

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

    def get_redq_q_target_no_grad(self, next_observations, rewards, done, ent_coef):
        # compute REDQ Q target, depending on the agent's Q target mode
        # allow min as a float:
        dyn_ent_weight = self.get_dynamics_entropy_weight()
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.action_log_prob(next_observations)
            next_log_prob = next_log_prob.reshape(-1, 1)
            next_features = self.extract_features(next_observations)
            # get entropy of transitions
            inp = th.cat([next_features, next_actions], dim=-1)
            inp = self.input_normalizer.normalize(inp)
            next_obs_dynamics_entropy = self.get_intrinsic_reward(
                inp=inp,
                labels=None,
            ).reshape(-1, 1)
            if self.q_target_mode == 'min':
                num_mins_to_use = get_probabilistic_num_min(self.num_min_critics)
                sample_idxs = np.random.choice(self.n_critics, num_mins_to_use, replace=False)
                """Q target is min of a subset of Q values"""

                q_prediction_next_cat = torch.cat(self.critic_target(next_observations,
                                                                      next_actions), 1)[..., sample_idxs]
                min_q, _ = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
                next_q_with_log_prob = min_q - ent_coef * (-next_obs_dynamics_entropy * dyn_ent_weight
                                                            + next_log_prob.reshape(-1, 1))
                y_q = rewards + self.gamma * (1 - done) * next_q_with_log_prob
            elif self.q_target_mode == 'ave':
                """Q target is average of all Q values"""
                q_values = th.cat(self.critic_target(
                    next_observations, next_actions), dim=1).mean(dim=1).reshape(-1, 1)
                next_q_with_log_prob = q_values - ent_coef * (-next_obs_dynamics_entropy * dyn_ent_weight
                                                            + next_log_prob.reshape(-1, 1))
                y_q = rewards + self.gamma * (1 - done) * next_q_with_log_prob
            else:
                raise NotImplementedError
        return y_q

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

        # Profile the training step
        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            if self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            """Q loss"""
            target_q_values = self.get_redq_q_target_no_grad(replay_data.next_observations,
                                                             replay_data.rewards,
                                                             replay_data.dones,
                                                             ent_coef=ent_coef,
                                                             )
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if ((gradient_step + 1) % self.policy_update_delay == 0) or gradient_step == gradient_steps - 1:
                # get policy loss
                actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
                log_prob = log_prob.reshape(-1, 1)
                # get dynamics entropy
                features = self.extract_features(replay_data.observations)
                inp = th.cat([features, actions_pi], dim=-1)
                inp = self.input_normalizer.normalize(inp)
                dynamics_entropy = self.get_intrinsic_reward(
                    inp=inp,
                    labels=None,
                ).reshape(-1, 1)
                batch_entropy = dynamics_entropy.mean()
                dynamics_info_gain.append(batch_entropy.item())
                target_info_gain.append(self.target_info_gain.item())
                batch_entropy = batch_entropy * dyn_ent_weight
                total_entropy = -log_prob + dynamics_entropy * dyn_ent_weight
                q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
                qf_pi = th.mean(q_values_pi, dim=1, keepdim=True)
                actor_loss = (-ent_coef * total_entropy - qf_pi).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                    # Important: detach the variable from the graph
                    # so we don't change it with other losses
                    # see https://github.com/rail-berkeley/softlearning/issues/60
                    ent_coef_loss = -(self.log_ent_coef * (-total_entropy +
                                                           self.target_info_gain + self.target_entropy).detach()).mean()
                    ent_coef_losses.append(ent_coef_loss.item())
                    self.ent_coef_optimizer.zero_grad()
                    ent_coef_loss.backward()
                    self.ent_coef_optimizer.step()

                self.target_info_gain = (1 - self.target_info_tau) * self.target_info_gain + \
                                        self.target_info_tau * batch_entropy.detach()

            if ((gradient_step + 1) % self.model_update_delay == 0) or gradient_step == gradient_steps - 1:
                # ensemble model training
                features = self.extract_features(replay_data.observations)
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

                    # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)


            # ensemble model training
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
        tb_log_name: str = "MaxEntDro",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        super().learn(
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
    from multimexmf.models.pretrain_models import DropoutEnsemble
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
        # 'train_freq': 32,
        # 'gradient_steps': 32,
        'learning_rate': 1e-3,
        'verbose': 1,
    }

    ensemble_model_kwargs = {
        'learn_std': False,
        'optimizer_kwargs': {'lr': 3e-4, 'weight_decay': 0.0},
        'features': (64, 64),
    }
    maximize_entropy = True
    if maximize_entropy:
        algorithm = MaxEntropyREDQ(
            env=vec_env,
            ensemble_type=DropoutEnsemble,
            ensemble_model_kwargs=ensemble_model_kwargs,
            model_update_delay=20,
            **algorithm_kwargs
        )
    else:
        algorithm = REDQ(
            env=vec_env,
            **algorithm_kwargs,
        )
    algorithm.learn(
        total_timesteps=50_000,
        # callback=eval_callback,
    )
