import collections
import copy

import torch
from stable_baselines3.ppo import PPO
from typing import Optional, Union, Dict, Type, Any, Tuple, List, ClassVar
from gymnasium import spaces
import numpy as np
import torch as th
from stable_baselines3.common.type_aliases import PyTorchObs
from multimexmf.models.pretrain_models import EnsembleMLP
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import Schedule, MaybeCallback
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from multimexmf.commons.buffers.replay_buffers import IntrinsicRewardRolloutBuffer
from stable_baselines3.common.vec_env import VecEnv
from torch.nn import functional as F
from stable_baselines3.common.utils import get_schedule_fn, explained_variance
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from multimexmf.commons.intrinsic_reward_algorithms.utils import BaseIntrinsicReward
from multimexmf.commons.intrinsic_reward_algorithms.utils import explore_till
from functools import partial
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN, CombinedExtractor


class Normalizer:
    def __init__(self, input_dim: int, update: bool = True):
        self.input_dim = input_dim
        self._reset_normalization_stats()
        self._update = update

    def reset(self):
        self._reset_normalization_stats()

    def _reset_normalization_stats(self):
        self.mean = np.zeros(self.input_dim)
        self.std = np.ones(self.input_dim)
        self.num_points = 0

    def update(self, x: np.ndarray):
        if not self._update:
            return
        assert len(x.shape) == 2 and x.shape[-1] == self.input_dim
        num_points = x.shape[0]
        total_points = num_points + self.num_points
        mean = (self.mean * self.num_points + np.sum(x, axis=0)) / total_points
        new_s_n = np.square(self.std) * self.num_points + np.sum(np.square(x - mean), axis=0) + \
                  self.num_points * np.square(self.mean - mean)

        new_var = new_s_n / total_points
        std = np.sqrt(new_var)
        self.mean = mean
        self.std = np.clip(std, a_min=1e-6, a_max=None)
        self.num_points = total_points

    def normalize(self, x: np.ndarray):
        return (x - self.mean) / self.std

    def denormalize(self, norm_x: np.ndarray):
        return norm_x * self.std + self.mean

    def scale(self, x: np.ndarray):
        return x / self.std


class IntrinsicActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(IntrinsicActorCriticPolicy, self).__init__(*args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        self.intrinsic_value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
                self.intrinsic_value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))
        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)  # type: ignore[call-arg]

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values, intrinsic_values = self.value_net(latent_vf), self.intrinsic_value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        # stack values and intrinsic values together
        values = th.cat([values, intrinsic_values], dim=-1)
        return actions, values, log_prob

    def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values, intrinsic_values = self.value_net(latent_vf), self.intrinsic_value_net(latent_vf)
        values = th.cat([values, intrinsic_values], dim=-1)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        values, intrinsic_values = self.value_net(latent_vf), self.intrinsic_value_net(latent_vf)
        return th.cat([values, intrinsic_values], dim=-1)


class IntrinsicActorCriticCnnPolicy(IntrinsicActorCriticPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            share_features_extractor: bool = True,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


class IntrinsicMultiInputActorCriticPolicy(IntrinsicActorCriticPolicy):
    def __init__(
            self,
            observation_space: spaces.Dict,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            share_features_extractor: bool = True,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


class IntrinsicPPO(PPO):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": IntrinsicActorCriticPolicy,
        "CnnPolicy": IntrinsicActorCriticCnnPolicy,
        "MultiInputPolicy": IntrinsicMultiInputActorCriticPolicy,
    }

    def __init__(self,
                 env: VecEnv,
                 policy: Type[IntrinsicActorCriticPolicy],
                 ensemble_model_kwargs: Dict,
                 intrinsic_model_batch_size: Optional[int] = None,
                 intrinsic_reward_weights: Optional[Dict] = None,
                 agg_intrinsic_reward: str = 'sum',
                 normalize_ensemble_training: bool = True,
                 pred_diff: bool = True,
                 intrinsic_reward_model: Optional[Type[BaseIntrinsicReward]] = None,
                 exploration_weight: Union[float, Schedule] = 0.5,
                 intrinsic_vf_coef: Optional[float] = None,
                 rollout_buffer_class: Optional[Type[IntrinsicRewardRolloutBuffer]] = IntrinsicRewardRolloutBuffer,
                 rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
                 normalize_intrinsic_reward: bool = False,
                 *args,
                 **kwargs
                 ):
        self.normalize_ensemble_training = normalize_ensemble_training
        self.normalize_intrinsic_reward = normalize_intrinsic_reward
        self.intrinsic_reward_weights = intrinsic_reward_weights
        self.agg_intrinsic_reward = agg_intrinsic_reward
        self.intrinsic_reward_model = intrinsic_reward_model
        self.exploration_weight = exploration_weight
        self.pred_diff = pred_diff

        super().__init__(env=env,
                         policy=policy,
                         rollout_buffer_class=rollout_buffer_class,
                         rollout_buffer_kwargs=rollout_buffer_kwargs,
                         *args, **kwargs)
        self._setup_ensemble_model(
            ensemble_model_kwargs=ensemble_model_kwargs,
            intrinsic_reward_weights=intrinsic_reward_weights,
            intrinsic_reward_model=intrinsic_reward_model,
            device=self.device,
            agg_intrinsic_reward=agg_intrinsic_reward,
        )
        self._setup_exploration_freq_schedule()
        if intrinsic_vf_coef:
            self.intrinsic_vf_coef = intrinsic_vf_coef
        else:
            self.intrinsic_vf_coef = self.vf_coef

        if intrinsic_model_batch_size:
            self.intrinsic_model_batch_size = intrinsic_model_batch_size
        else:
            self.intrinsic_model_batch_size = self.batch_size

    def _setup_model(self):
        super()._setup_model()
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
        label_dict = self._get_ensemble_targets(sample_obs,
                                                sample_obs,
                                                rewards=th.zeros((1, 1))
                                                )

        label_dict = {key: val.shape[1:] for key, val in label_dict.items()}
        self.rollout_buffer.setup_intrinsic_model_buffer(input_dim, label_dict)

    def _setup_exploration_freq_schedule(self) -> None:
        """Transform to callable if needed."""
        self.exploration_weight_fn = get_schedule_fn(self.exploration_weight)

    def _update_exploration_weight(self) -> None:
        self.exploration_weight = self.exploration_weight_fn(self._current_progress_remaining)

    def _setup_normalizer(self, input_dim: int, output_dict: Dict, device: th.device):
        self.input_normalizer = Normalizer(input_dim=input_dim, update=self.normalize_ensemble_training)
        output_normalizers = {}
        for key, val in output_dict.items():
            output_normalizers[key] = Normalizer(input_dim=val.shape[-1], update=self.normalize_ensemble_training)
        self.output_normalizers = output_normalizers
        self.int_reward_normalizer = Normalizer(input_dim=1, update=True)

    def _get_ensemble_inputs(self, observation: np.ndarray, action: np.ndarray):
        feat = self.extract_features(
            obs_as_tensor(observation,
                          self.device)
        ).clone().cpu().numpy()
        inp = np.concatenate([feat, action], axis=-1)
        return inp

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
                                                 rewards=np.zeros((1, 1))
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
            features = self.policy.extract_features(
                obs, features_extractor=self.policy.features_extractor)
        return features

    def _get_ensemble_targets(self, next_obs: Union[np.ndarray, Dict],
                              obs: Union[np.ndarray, Dict],
                              rewards: np.ndarray) -> Dict:
        if self.pred_diff:
            next_obs = obs_as_tensor(next_obs, self.device)
            obs = obs_as_tensor(obs, self.device)
            next_obs = self.extract_features(next_obs).clone().cpu().numpy()
            obs = self.extract_features(obs).clone().cpu().numpy()
            return {
                'next_obs': next_obs - obs,
                'reward': rewards.reshape(-1, 1),
            }
        else:
            next_obs = obs_as_tensor(next_obs, self.device)
            next_obs = self.extract_features(next_obs).clone().cpu().numpy()
            return {
                'next_obs': next_obs,
                'reward': rewards.reshape(-1, 1),
            }

    def get_intrinsic_reward(self, inp: np.ndarray, labels: Dict) -> th.Tensor:
        # calculate intrinsic reward
        inp = torch.Tensor(inp, device=self.device)
        torch_labels = {key: torch.Tensor(val, device=self.device) for key, val in labels.items()}
        if self.intrinsic_reward_model is None:
            return th.zeros(inp.shape[0], device=self.device)
        else:
            return self.intrinsic_reward_model(inp=inp, labels=torch_labels).clone().cpu().numpy()

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # update exploration weight
        self._update_exploration_weight()
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses, intrinsic_value_losses = [], [], []
        clip_fractions = []

        ensemble_losses = collections.defaultdict(list)

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values, intrinsic_values = values[..., 0], values[..., -1]
                # Normalize advantage
                advantages = rollout_data.advantages
                intrinsic_advantages = rollout_data.intrinsic_advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    intrinsic_advantages = (intrinsic_advantages - intrinsic_advantages.mean()) / \
                                           (intrinsic_advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                total_advantage = advantages + self.exploration_weight * intrinsic_advantages
                # clipped surrogate loss
                policy_loss_1 = total_advantage * ratio
                policy_loss_2 = total_advantage * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                    intrinsic_values_pred = intrinsic_values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                    intrinsic_values_pred = rollout_data.old_intrinsic_value + th.clamp(
                        intrinsic_values - rollout_data.old_intrinsic_value, -clip_range_vf, clip_range_vf
                    )

                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                intrinsic_value_loss = F.mse_loss(rollout_data.intrinsic_returns, intrinsic_values_pred)
                intrinsic_value_losses.append(intrinsic_value_loss.item())
                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + \
                       self.intrinsic_vf_coef * intrinsic_value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                # training model for intrinsic rewards
                intrinsic_model_inp, intrinsic_model_labels = self.rollout_buffer.intrinsic_model_samples(
                    self.intrinsic_model_batch_size
                )
                self.ensemble_model.optimizer.zero_grad()
                prediction = self.ensemble_model(intrinsic_model_inp)
                ensemble_loss = self.ensemble_model.loss(prediction=prediction, target=intrinsic_model_labels)
                stacked_losses = []
                for key, val in ensemble_loss.items():
                    ensemble_losses[key].append(val.item())
                    stacked_losses.append(val)
                stacked_losses = th.stack(stacked_losses)
                total_loss = stacked_losses.mean()
                total_loss.backward()
                self.ensemble_model.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(),
                                           self.rollout_buffer.returns.flatten())

        explained_intrinsic_var = explained_variance(self.rollout_buffer.intrinsic_values.flatten(),
                                                     self.rollout_buffer.intrinsic_returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/intrinsic_value_loss", np.mean(intrinsic_value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/explained_intrinsic_variance", explained_intrinsic_var)
        self.logger.record("train/exploration_weight", self.exploration_weight)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        for key, val in ensemble_losses.items():
            self.logger.record(f"train/ensemble_loss_{key}", np.mean(val))
            self.logger.record(f"train/out_normalizer_mean_{key}", np.mean(
                self.output_normalizers[key].mean))
            self.logger.record(f"train/out_normalizer_std_{key}", np.mean(
                self.output_normalizers[key].std))
        self.logger.record(f"train/inp_normalizer_mean", np.mean(self.input_normalizer.mean))
        self.logger.record(f"train/inp_normalizer_std", np.mean(self.input_normalizer.std))
        self.logger.record(f"train/intrinsic_reward_normalizer_mean", np.mean(self.int_reward_normalizer.mean))
        self.logger.record(f"train/intrinsic_reward_normalizer_std", np.mean(self.int_reward_normalizer.std))

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: IntrinsicRewardRolloutBuffer,
            n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        numpy_obs = True
        if isinstance(self.observation_space.sample(), np.ndarray):
            obs_ = []
            next_obs_ = []
        elif isinstance(self.observation_space.sample(), Dict):
            obs_ = collections.defaultdict(list)
            next_obs_ = collections.defaultdict(list)
            numpy_obs = False
        else:
            raise NotImplementedError
        rewards_ = []
        acts = []
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

            self.num_timesteps += env.num_envs

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
            next_obs = copy.deepcopy(new_obs)
            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                ):
                    term_obs = copy.deepcopy(infos[idx]["terminal_observation"])
                    if numpy_obs:
                        next_obs[idx] = np.array(infos[idx]["terminal_observation"])
                    else:
                        for key, val in term_obs.items():
                            next_obs[key][idx] = np.array(val)
                    if infos[idx].get("TimeLimit.truncated", False):
                        terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]

                        with th.no_grad():
                            terminal_value = self.policy.predict_values(terminal_obs)  # type: ignore[arg-type]
                        rewards[idx] += self.gamma * terminal_value[..., 0]
                    # TODO: See if this correction is needed for intrinsic reward since it is nonepisodic
                    # intrinsic_rewards[idx] *= self.gamma * terminal_value[..., -1]

            # rewards = np.concatenate([rewards.reshape(-1, 1), intrinsic_rewards.clone().detach().numpy()], axis=-1)
            rollout_buffer.add(
                obs=self._last_obs,
                action=actions,
                reward=rewards,
                episode_start=self._last_episode_starts,
                value=values,
                log_prob=log_probs,
            )
            if numpy_obs:
                obs_.append(self._last_obs)
                next_obs_.append(next_obs)
            else:
                for key in self._last_obs.keys():
                    obs_[key].append(self._last_obs[key])
                    next_obs_[key].append(next_obs[key])
            acts.append(clipped_actions)
            rewards_.append(rewards)

            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

            # rollout_buffer.add_intrinsic_model_data(
            #    inp=inp,
            #    labels=labels,
            # )

        with th.no_grad():
            if numpy_obs:
                next_obs = np.vstack(next_obs_)
                obs = np.vstack(obs_)
            else:
                next_obs = {key: np.vstack(val) for key, val in next_obs_.items()}
                obs = {key: np.vstack(val) for key, val in obs_.items()}
            rewards = np.hstack(rewards_)
            acts = np.vstack(acts)
            labels = self._get_ensemble_targets(
                obs=obs,
                next_obs=next_obs,
                rewards=rewards,
            )
            inp = self._get_ensemble_inputs(obs, acts)
            self.input_normalizer.update(inp)
            inp = self.input_normalizer.normalize(inp)
            for key, y in labels.items():
                # if gradient_step == normalization_index:
                self.output_normalizers[key].update(y)
                labels[key] = self.output_normalizers[key].normalize(y)
            intrinsic_rewards = self.get_intrinsic_reward(labels=labels, inp=inp).reshape(-1, 1)
            # self.int_reward_normalizer.update(intrinsic_rewards)
            # TODO: See if normalization is required
            if self.normalize_intrinsic_reward:
                intrinsic_rewards = self.int_reward_normalizer.scale(intrinsic_rewards)
            # intrinsic_rewards = self.int_reward_normalizer.scale(intrinsic_rewards)
            # reshape everything back:
            intrinsic_rewards = intrinsic_rewards.reshape(-1, self.n_envs)
            inp = inp.reshape((-1, self.n_envs) + inp.shape[1:])
            labels = {key: val.reshape((-1, self.n_envs) + val.shape[1:]) for key, val in labels.items()}
            rollout_buffer.add_batch_intrinsic_rewards(intrinsic_rewards)
            rollout_buffer.add_intrinsic_model_data(
                inp=inp,
                labels=labels,
            )

            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        # update intrinsic normalization by intrinsic returns
        self.int_reward_normalizer.update(rollout_buffer.intrinsic_returns.reshape(-1, 1))
        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            tb_log_name: str = "IntrinsicPPO",
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
    from multimexmf.commons.intrinsic_reward_algorithms.utils import \
        DisagreementIntrinsicReward, exploration_frequency_schedule
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


    log_dir = './check/'
    n_envs = 4
    vec_env = make_vec_env(CustomPendulumEnv, n_envs=4, seed=0, wrapper_class=TimeLimit,
                           env_kwargs={'render_mode': 'rgb_array'},
                           wrapper_kwargs={'max_episode_steps': 200})
    # vec_env = VecVideoRecorder(venv=vec_env,
    #                           video_folder=log_dir,
    #                           record_video_trigger=lambda x: True,
    #                          )

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
        # VecVideoRecorder(
        make_vec_env(CustomPendulumEnv, n_envs=4, seed=1,
                     env_kwargs={'render_mode': 'rgb_array'},
                     wrapper_class=TimeLimit,
                     wrapper_kwargs={'max_episode_steps': 200}
                     ),
        #   video_folder=log_dir + 'eval/',
        #   record_video_trigger=lambda x: True,
        # ),
        log_path=log_dir,
        best_model_save_path=log_dir,
        eval_freq=1000,
        n_eval_episodes=5,
        deterministic=True,
    )
    algorithm_kwargs = {
        # 'train_freq': 32,
        # 'gradient_steps': 32,
        'policy': 'MlpPolicy',
        'verbose': 1,
        'n_steps': 1024,
        'gae_lambda': 0.95,
        'gamma': 0.9,
        'n_epochs': 10,
        'learning_rate': 1e-3,
        'clip_range': 0.2,
        'ent_coef': 0.0,
        'use_sde': True,
        'sde_sample_freq': 4,
    }

    ensemble_model_kwargs = {
        'learn_std': False,
        'optimizer_kwargs': {'lr': 1e-4},
        'use_entropy': False,
    }
    intrinsic = True
    if intrinsic:
        schedule = explore_till(exploration_till_progress=0.75)
        exploration_weight = lambda progress: 0.5 * schedule(progress=progress)
        algorithm = IntrinsicPPO(
            env=vec_env,
            ensemble_model_kwargs=ensemble_model_kwargs,
            intrinsic_reward_model=DisagreementIntrinsicReward,
            exploration_weight=exploration_weight,
            **algorithm_kwargs
        )
    else:
        algorithm = PPO(
            env=vec_env,
            **algorithm_kwargs
        )
    algorithm.learn(
        total_timesteps=250_000,
        callback=eval_callback,
    )
