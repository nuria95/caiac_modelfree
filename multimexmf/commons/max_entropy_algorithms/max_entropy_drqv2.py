import collections
from multimexmf.commons.drqv2 import DrQv2, DrQv2Policy, clamp
from multimexmf.commons.intrinsic_reward_algorithms.drq_exploit_and_play import Pooling
from typing import Optional, Union, Dict, Type
from stable_baselines3.common.type_aliases import Schedule, MaybeCallback
import torch as th
from multimexmf.models.pretrain_models import EnsembleMLP, Normalizer, EPS
from stable_baselines3.common.utils import polyak_update, get_parameters_by_name
import numpy as np
import copy
import torch.nn.functional as F
from stable_baselines3.common.type_aliases import TensorDict
from multimexmf.commons.intrinsic_reward_algorithms.utils import DisagreementIntrinsicReward


class MaxEntDrQv2Policy(DrQv2Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule=lr_schedule)
        self.exploration_critic = self.make_critic(features_extractor=None)
        self.exploration_critic_target = self.make_critic(features_extractor=None)
        self.exploration_critic_target.load_state_dict(self.exploration_critic.state_dict())
        self.exploration_critic.optimizer = self.optimizer_class(
            self.exploration_critic.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )
        self.exploration_critic_target.set_training_mode(False)

    def set_training_mode(self, mode: bool) -> None:
        super().set_training_mode(mode)
        self.exploration_critic.set_training_mode(mode)


class MaxEntDrQv2(DrQv2):
    def __init__(self,
                 ensemble_model_kwargs: Dict,
                 policy: Type[MaxEntDrQv2Policy] = MaxEntDrQv2Policy,
                 intrinsic_reward_weights: Optional[Dict] = None,
                 agg_intrinsic_reward: str = 'mean',
                 normalize_ensemble_training: bool = True,
                 normalize_dyn_entropy: bool = True,
                 ensemble_type: Type[th.nn.Module] = EnsembleMLP,
                 pred_diff: bool = False,
                 predict_img_embed: bool = False,
                 predict_tactile_embed: bool = True,
                 ens_obs_representation: str = 'full',
                 predict_state: bool = True,
                 predict_reward: bool = True,
                 update_encoder_with_exploration_critic: bool = True,
                 train_ensemble_with_target: bool = False,
                 img_obs_pooling_params: Optional[dict] = None,
                 tactile_obs_pooling_params: Optional[dict] = None,
                 dyn_entropy_scale: Union[str, float] = 'auto',
                 init_dyn_entropy_scale: float = 1,
                 use_optimism: bool = False,
                 *args,
                 **kwargs
                 ):
        self.dyn_entropy_scale = dyn_entropy_scale
        self.init_dyn_entropy_scale = init_dyn_entropy_scale
        super().__init__(policy=policy, *args, **kwargs)
        self.ens_obs_representation = ens_obs_representation
        self.normalize_ensemble_training = normalize_ensemble_training
        self.normalize_dyn_entropy = normalize_dyn_entropy
        self.pred_diff = pred_diff
        self.predict_img_embed = predict_img_embed
        self.predict_tactile_embed = predict_tactile_embed
        self.predict_state = predict_state
        self.predict_reward = predict_reward
        self.train_ensemble_with_target = train_ensemble_with_target
        self.img_obs_pooling_params = img_obs_pooling_params
        self.tactile_obs_pooling_params = tactile_obs_pooling_params
        self.use_optimism = use_optimism
        self.update_encoder_with_exploration_critic = update_encoder_with_exploration_critic
        self._setup_ensemble_model(
            ensemble_type=ensemble_type,
            ensemble_model_kwargs=ensemble_model_kwargs,
            intrinsic_reward_weights=intrinsic_reward_weights,
            device=self.device,
            agg_intrinsic_reward=agg_intrinsic_reward,
        )

    def _setup_model(self) -> None:
        super()._setup_model()
        self.exploration_critic_batch_norm_stats = get_parameters_by_name(self.exploration_critic, ["running_"])
        self.exploration_critic_batch_norm_stats_target = get_parameters_by_name(self.exploration_critic_target,
                                                                                 ["running_"])

        if self.dyn_entropy_scale == 'auto':
            assert self.init_dyn_entropy_scale > 0
            # we initialize the weight such that policy entropy and information gain are of similar orders.
            self.log_dyn_entropy_scale = th.log(th.ones(1, device=self.device)
                                                * self.init_dyn_entropy_scale).requires_grad_(True)
            self.dyn_ent_scale_optimizer = th.optim.Adam([self.log_dyn_entropy_scale],
                                                         lr=self.lr_schedule(1))
            self.dyn_entropy_scale = None
        else:
            self.dyn_entropy_scale = th.tensor([self.dyn_entropy_scale]).squeeze()
            self.dyn_ent_scale_optimizer = None
            self.log_dyn_entropy_scale = None

    def _create_aliases(self) -> None:
        super()._create_aliases()
        self.exploration_critic = self.policy.exploration_critic
        self.exploration_critic_target = self.policy.exploration_critic_target

    def _extract_obs_representation_for_ensemble(self, obs):
        if self.ens_obs_representation == 'full':
            # we do augmentation of obs to enforce invariances
            return self.encode_observation(observation=obs, detach=True, augment=True)
        elif self.ens_obs_representation == 'repeat':
            # repeats the last obs and returns the embedding for the resulting image/observation
            # currently assumes we have only 3 channels
            extracted_obs = copy.deepcopy(obs)

            def repeat_final_obs(x):
                assert len(x.shape) == 4
                n_stacks = x.shape[1] // 3
                final_obs = x[:, -3:, ...]
                for i in range(n_stacks - 1):
                    x[:, i * 3: (i + 1) * 3, ...] = final_obs
                return x

            if self.policy.img_encoder is not None:
                extracted_obs[self.policy.image_key] = repeat_final_obs(obs[self.policy.image_key])
            if self.policy.tactile_encoder is not None:
                extracted_obs['tactile'] = repeat_final_obs(obs['tactile'])
            return self.encode_observation(observation=extracted_obs, detach=True, augment=True)
        else:
            raise NotImplementedError

    def _setup_normalizer(self, input_dim: int, output_dict: Dict, device: th.device):
        self.input_normalizer = Normalizer(input_dim=input_dim, update=self.normalize_ensemble_training,
                                           device=device)
        output_normalizers = {}
        for key, val in output_dict.items():
            output_normalizers[key] = Normalizer(input_dim=val.shape[-1], update=self.normalize_ensemble_training,
                                                 device=device)
        self.output_normalizers = output_normalizers
        self.dyn_entropy_normalizer = Normalizer(input_dim=1, update=self.normalize_dyn_entropy,
                                                 device=device)

    def _setup_ensemble_model(self,
                              ensemble_model_kwargs: Dict,
                              intrinsic_reward_weights: Dict,
                              device: th.device,
                              ensemble_type: Type[th.nn.Module] = EnsembleMLP,
                              agg_intrinsic_reward: str = 'mean',
                              ) -> None:
        # setup encoders for observations that give target embeddings for the ensemble to learn
        self._setup_encoders_for_ensemble_targets()

        # sample random obs and set up ensmeble model
        sample_obs = self.observation_space.sample()
        for key in sample_obs.keys():
            sample_obs[key] = np.expand_dims(sample_obs[key], 0)
        input_dim = self.critic.features_extractor.features_dim + self.action_space.shape[0]
        sample_obs, _ = self.policy.obs_to_tensor(sample_obs)
        sample_state = self._get_state_from_observation(observation=sample_obs, detach=True, augment=True)

        output_dict = self._get_ensemble_targets(obs=sample_obs,
                                                 state=sample_state,
                                                 next_state=sample_state,
                                                 reward=th.zeros((1, 1))
                                                 )
        self.ensemble_model = ensemble_type(
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

        self.intrinsic_reward_model = DisagreementIntrinsicReward(
            intrinsic_reward_weights=intrinsic_reward_weights,
            ensemble_model=self.ensemble_model,
            agg_intrinsic_reward=agg_intrinsic_reward,
        )

    def _setup_encoders_for_ensemble_targets(self):
        self._has_encoders_for_ensemble_targets = False
        if self.policy.img_encoder is not None:
            if self.predict_img_embed:
                if self.img_obs_pooling_params is not None:
                    self.img_target_encoder = Pooling(**self.img_obs_pooling_params)
                else:
                    self.img_target_encoder = Pooling()
                self.img_target_encoder.to(self.device)
                self._has_encoders_for_ensemble_targets = True

        if self.policy.tactile_encoder is not None:
            if self.predict_tactile_embed:
                if self.tactile_obs_pooling_params is not None:
                    self.tactile_target_encoder = Pooling(**self.tactile_obs_pooling_params)
                else:
                    self.tactile_target_encoder = Pooling()
                self.tactile_target_encoder.to(self.device)
                self._has_encoders_for_ensemble_targets = True

    @property
    def has_encoders_for_ensemble_targets(self):
        return self._has_encoders_for_ensemble_targets

    def _get_ensemble_targets(self,
                              obs: TensorDict,
                              state: th.Tensor,
                              next_state: th.Tensor,
                              reward: th.Tensor,
                              ):
        targets = {}
        if self.policy.img_encoder is not None and self.policy.tactile_encoder is not None:
            # extract CNN output
            embed_obs = self._extract_obs_representation_for_ensemble(obs)
            img_obs, tactile_obs = embed_obs[..., :self.policy.img_encoder.repr_dim], \
                embed_obs[..., self.policy.img_encoder.repr_dim: self.policy.img_encoder.repr_dim
                                                                 + self.policy.tactile_encoder.repr_dim]
            # get encoding of CNN output
            if self.predict_img_embed:
                assert self.has_encoders_for_ensemble_targets
                targets[self.policy.image_key] = self.img_target_encoder(img_obs)
            if self.predict_tactile_embed:
                assert self.has_encoders_for_ensemble_targets
                targets['tactile'] = self.tactile_target_encoder(tactile_obs)
        elif self.policy.img_encoder is not None:
            if self.predict_img_embed:
                assert self.has_encoders_for_ensemble_targets
                embed_obs = self._extract_obs_representation_for_ensemble(obs)
                # targets[self.policy.image_key] = self.img_target_encoder.encode(
                #    embed_obs[..., :self.policy.img_encoder.repr_dim])
                targets[self.policy.image_key] = self.img_target_encoder(
                    embed_obs[..., :self.policy.img_encoder.repr_dim])
        elif self.policy.tactile_encoder is not None:
            if self.predict_tactile_embed:
                assert self.has_encoders_for_ensemble_targets
                embed_obs = self._extract_obs_representation_for_ensemble(obs)
                targets['tactile'] = self.tactile_target_encoder(
                    embed_obs[..., :self.policy.tactile_encoder.repr_dim]
                )
        else:
            raise NotImplementedError
        if self.predict_state:
            if self.pred_diff:
                targets['next_state'] = next_state - state
            else:
                targets['next_state'] = next_state
        if self.predict_reward:
            targets['reward'] = reward
        return targets

    def _get_state_from_embedded_obs(self, embed: th.Tensor, use_target_critic: bool = False):
        if use_target_critic:
            return self.critic_target.extract_features(embed).detach()
        else:
            return self.critic.extract_features(embed).detach()

    def _get_state_from_observation(self, observation, detach: bool = False, augment: bool = True):
        obs = self.encode_observation(observation=observation, detach=detach, augment=augment)
        return self._get_state_from_embedded_obs(obs, use_target_critic=self.train_ensemble_with_target)

    def get_intrinsic_reward(self, inp: th.Tensor, labels: Dict) -> th.Tensor:
        # calculate intrinsic reward
        entropy = self.intrinsic_reward_model(inp=inp, labels=labels)
        if not self.ensemble_model.learn_std:
            info_gain = entropy - th.log(th.ones_like(entropy) * EPS)
            return info_gain
        else:
            return entropy

    def get_actor_and_target_entropy(self, obs):
        with th.no_grad():
            noise_std, noise_clip = self.noise_std
            actions_actor = self.actor(obs)
            noise = actions_actor.clone().data.normal_(0, noise_std)
            noise = clamp(noise, noise_clip)
            actions_actor = clamp(actions_actor + noise, 1.0)

            actions_actor_target = self.actor_target(obs)
            noise = actions_actor_target.clone().data.normal_(0, noise_std)
            noise = clamp(noise, noise_clip)
            actions_actor_target = clamp(actions_actor_target + noise, 1.0)

            state = self._get_state_from_embedded_obs(obs,
                                                      use_target_critic=self.train_ensemble_with_target)

            inp = th.cat([state, actions_actor], dim=-1)
            target_inp = th.cat([state, actions_actor_target], dim=-1)
            total_inp = th.cat([inp, target_inp], dim=0)
            total_inp = self.input_normalizer.normalize(total_inp)
            dynamics_entropy = self.get_intrinsic_reward(
                inp=total_inp,
                labels=None,
            ).reshape(-1, 1)
        return dynamics_entropy

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        self.ensemble_model.train(True)
        # Update learning rate according to lr schedule
        optimizers = [self.actor.optimizer, self.critic.optimizer, self.exploration_critic.optimizer]
        if self.dyn_ent_scale_optimizer is not None:
            optimizers += [self.dyn_ent_scale_optimizer]
        self._update_learning_rate(optimizers)
        self._update_encoder_learning_rate()

        actor_losses, critic_losses, exploration_critic_losses = [], [], []
        dynamics_entropies, target_entropies = [], []
        dyn_scales, dyn_scale_losses = [], []

        ensemble_losses = collections.defaultdict(list)

        self._update_ensemble_normalizers(batch_size)

        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env,
                                                    n_steps=self.n_steps)  # type: ignore[union-attr]

            obs = self.encode_observation(observation=replay_data.observations, detach=False, augment=True)

            dynamics_entropy = self.get_actor_and_target_entropy(obs)
            self.dyn_entropy_normalizer.update(dynamics_entropy.detach())
            dynamics_entropy = self.dyn_entropy_normalizer.normalize(dynamics_entropy)
            dynamics_entropy, target_dynamics_entropy = dynamics_entropy[:batch_size], \
                dynamics_entropy[batch_size:]
            dynamics_entropies.append(dynamics_entropy.mean().detach().item())
            target_entropies.append(target_dynamics_entropy.mean().item())

            if self.dyn_ent_scale_optimizer is not None and self.log_dyn_entropy_scale is not None:
                dyn_scale = th.exp(self.log_dyn_entropy_scale.detach())
                dyn_scale_loss = (self.log_dyn_entropy_scale * (
                        dynamics_entropy - target_dynamics_entropy).detach()).mean()
                dyn_scale_losses.append(dyn_scale_loss.item())
                self.dyn_ent_scale_optimizer.zero_grad()
                dyn_scale_loss.backward()
                self.dyn_ent_scale_optimizer.step()
            else:
                dyn_scale = self.dyn_entropy_scale

            dyn_scales.append(dyn_scale.item())

            # sample data with 1 step transition to train dynamics entropy critic
            expl_replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            expl_obs = self.encode_observation(observation=expl_replay_data.observations,
                                               detach=bool(1 - self.update_encoder_with_exploration_critic),
                                               augment=True)
            with th.no_grad():
                next_obs = self.encode_observation(observation=replay_data.next_observations, detach=True, augment=True)
                # Select action according to policy and add clipped noise
                noise_std, noise_clip = self.noise_std
                noise = replay_data.actions.clone().data.normal_(0, noise_std)
                noise = clamp(noise, noise_clip)

                # We use actor (not actor target here) for the q value update
                next_actions = clamp(self.actor(next_obs) + noise, 1.0)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(next_obs, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                # estimate target value for exploration critic (only using 1 step return)
                expl_next_obs = self.encode_observation(observation=expl_replay_data.next_observations,
                                                        detach=True, augment=True)
                expl_state = self._get_state_from_embedded_obs(expl_obs,
                                                               use_target_critic=self.train_ensemble_with_target)

                inp = th.cat([expl_state, expl_replay_data.actions], dim=-1)
                inp = self.input_normalizer.normalize(inp)
                rewards = self.get_intrinsic_reward(
                    inp=inp,
                    labels=None
                ).reshape(-1, 1)
                rewards = self.dyn_entropy_normalizer.normalize(rewards)
                noise = expl_replay_data.actions.clone().data.normal_(0, noise_std)
                noise = clamp(noise, noise_clip)

                # We use actor (not actor target here) for the q value update
                expl_next_actions = clamp(self.actor(expl_next_obs) + noise, 1.0)
                expl_next_q_values = th.cat(self.exploration_critic_target(expl_next_obs, expl_next_actions), dim=1)
                expl_next_q_values, _ = th.min(expl_next_q_values, dim=1, keepdim=True)
                # not using done flag for exploration critic as proposed in RND, also we only consider 1 step return
                # for intrinsic reward
                expl_target_q_values = rewards + self.gamma * expl_next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(obs, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            expl_q_values = self.exploration_critic(expl_obs, expl_replay_data.actions)
            expl_critic_loss = sum(F.mse_loss(current_q, expl_target_q_values) for current_q in expl_q_values)
            assert isinstance(expl_critic_loss, th.Tensor)
            exploration_critic_losses.append(expl_critic_loss.item())

            critic_loss = expl_critic_loss + critic_loss
            # Optimize the critics
            self.policy.encoder_optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            self.exploration_critic.optimizer.zero_grad()
            critic_loss.backward()
            self.exploration_critic.optimizer.step()
            self.critic.optimizer.step()
            self.policy.encoder_optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                obs = obs.detach()
                noise = replay_data.actions.clone().data.normal_(0, noise_std)
                noise = clamp(noise, noise_clip)
                actions = clamp(self.actor(obs) + noise, 1.0)
                q_values = th.cat(self.critic(obs, actions), dim=1)
                expl_q_values = th.cat(self.exploration_critic(obs, actions), dim=1)
                if self.use_optimism:
                    q_values, _ = th.max(q_values, dim=1, keepdim=True)
                    expl_q_values, _ = th.max(expl_q_values, dim=1, keepdim=True)
                else:
                    q_values, _ = th.min(q_values, dim=1, keepdim=True)
                    expl_q_values, _ = th.min(expl_q_values, dim=1, keepdim=True)
                total_q_values = q_values + dyn_scale * expl_q_values
                actor_loss = - total_q_values.mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

                # ensemble model training
                inp = th.cat([expl_state, expl_replay_data.actions], dim=-1)
                inp = self.input_normalizer.normalize(inp)

                expl_next_state = self._get_state_from_embedded_obs(expl_next_obs,
                                                                    use_target_critic=self.train_ensemble_with_target)
                labels = self._get_ensemble_targets(
                    obs=expl_replay_data.observations,
                    state=expl_state,
                    next_state=expl_next_state,
                    reward=expl_replay_data.rewards)
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

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        if len(dyn_scale_losses) > 0:
            self.logger.record("train/dyn_scale_loss", np.mean(dyn_scale_losses))
        if len(dynamics_entropies) > 0:
            self.logger.record("train/dynamics_entropy", np.mean(dynamics_entropies))
            self.logger.record("train/target_entropy", np.mean(target_entropies))
        self.logger.record("train/dyn_scale", np.mean(dyn_scales))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/exploration_critic_loss", np.mean(exploration_critic_losses))
        for key, val in ensemble_losses.items():
            self.logger.record(f"train/ensemble_loss_{key}", np.mean(val))
            self.logger.record(f"train/out_normalizer_mean_{key}", np.mean(
                self.output_normalizers[key].mean.cpu().numpy()))
            self.logger.record(f"train/out_normalizer_std_{key}", np.mean(
                self.output_normalizers[key].std.cpu().numpy()))
        self.logger.record("train/inp_normalizer_mean", np.mean(self.input_normalizer.mean.cpu().numpy()))
        self.logger.record("train/inp_normalizer_std", np.mean(self.input_normalizer.std.cpu().numpy()))
        self.logger.record("train/dynamics_entropy_mean", np.mean(self.dyn_entropy_normalizer.mean.cpu().numpy()))
        self.logger.record("train/dynamics_entropy_std", np.mean(self.dyn_entropy_normalizer.std.cpu().numpy()))
        noise, _ = self.noise_std
        self.logger.record("train/noise", noise)
        self.ensemble_model.train(False)

    def _update_ensemble_normalizers(self, batch_size: int):
        if self.replay_buffer.size() >= batch_size:
            replay_data = self.replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)
            state = self._get_state_from_observation(observation=replay_data.observations, detach=True, augment=False)
            next_state = self._get_state_from_observation(observation=replay_data.next_observations,
                                                          detach=True, augment=False)
            inp = th.cat([state, replay_data.actions], dim=-1)
            self.input_normalizer.update(inp.detach())

            labels = self._get_ensemble_targets(
                obs=replay_data.observations,
                state=state,
                next_state=next_state,
                reward=replay_data.rewards)

            for key, y in labels.items():
                self.output_normalizers[key].update(y.detach())

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            tb_log_name: str = "MaxEntDrQv2",
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
    from gymnasium.wrappers.time_limit import TimeLimit
    from gymnasium.envs.mujoco.reacher_v4 import ReacherEnv
    from gymnasium.wrappers.pixel_observation import PixelObservationWrapper
    from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack

    env = lambda: PixelObservationWrapper(TimeLimit(ReacherEnv(render_mode='rgb_array',
                                                               height=84,
                                                               width=84), max_episode_steps=50))
    print('using image observation')

    vec_env = VecFrameStack(make_vec_env(env, n_envs=4, seed=0), n_stack=3)

    algorithm_kwargs = {
        'learning_rate': 1e-4,
        'verbose': 1,
        # 'learning_starts': 1000,
        # 'tensorboard_log': "./logs/",
    }

    ensemble_model_kwargs = {
        'learn_std': False,
        'optimizer_kwargs': {'lr': 1e-4}
    }
    algorithm = MaxEntDrQv2(
        ensemble_model_kwargs=ensemble_model_kwargs,
        env=vec_env,
        seed=0,
        buffer_size=100_000,
        predict_img_embed=True,
        **algorithm_kwargs,
    )

    algorithm.learn(
        total_timesteps=1000,
        log_interval=1,
    )
