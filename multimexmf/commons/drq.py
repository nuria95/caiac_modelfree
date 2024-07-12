import numpy as np
import torch
import torch.nn as nn
from multimexmf.models.encoder_decoder_models import weight_init
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.sac import SAC
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Type, Union
import torch as th
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import polyak_update, get_schedule_fn, update_learning_rate
import torch.nn.functional as F
from stable_baselines3.common.type_aliases import MaybeCallback, Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.sac.policies import Actor
from stable_baselines3.common.policies import ContinuousCritic


# augmentation module for images taken from https://github.com/facebookresearch/drqv2/blob/main/drqv2.py
class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.reshape(h.shape[0], -1)
        return h


class Trunk(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Space,
                 feature_dim: int, has_image: bool,
                 has_state: bool,
                 has_tactile: bool, img_repr_dim: int = -1,
                 tactile_repr_dim: int = -1,
                 state_dim: int = -1,
                 act_fn: nn.Module = nn.ReLU):

        total_feature_dim = feature_dim
        if has_state:
            assert state_dim > 0
            total_feature_dim += state_dim

        super().__init__(observation_space=observation_space, features_dim=total_feature_dim)

        if has_image and has_tactile:
            assert tactile_repr_dim > 0 and img_repr_dim > 0
            # add a module which combines these 2 modalities
            self.trunk = nn.Sequential(
                nn.Linear(img_repr_dim + tactile_repr_dim + has_state * state_dim, 2 * feature_dim),
                act_fn(),
                nn.Linear(2 * feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.Tanh()
            )

        elif has_tactile:
            assert tactile_repr_dim > 0
            self.trunk = nn.Sequential(nn.Linear(tactile_repr_dim + has_state * state_dim, feature_dim),
                                       nn.LayerNorm(feature_dim), nn.Tanh())

        elif has_image:
            assert img_repr_dim > 0
            self.trunk = nn.Sequential(nn.Linear(img_repr_dim + has_state * state_dim, feature_dim),
                                       nn.LayerNorm(feature_dim), nn.Tanh())
        else:
            raise NotImplementedError

        self.apply(weight_init)
        self.has_image = has_image
        self.has_tactile = has_tactile
        self.has_state = has_state
        self.state_dim = state_dim

    def forward(self, observations: torch.Tensor):
        """
        :param observations: concatenated output of the encoder block(s)
        :return:
        """
        latent = self.trunk(observations)
        if self.has_state:
            # State has to be stored towards the end of the embedding
            state = observations[..., :-self.state_dim]
            latent = torch.cat([latent, state], dim=-1)
        return latent


class DrQActor(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply(weight_init)

    def extract_features(self, obs: PyTorchObs, features_extractor: BaseFeaturesExtractor) -> th.Tensor:
        return self.features_extractor(obs)


class DrQCritic(ContinuousCritic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply(weight_init)

    def extract_features(self, obs: PyTorchObs, features_extractor: BaseFeaturesExtractor) -> th.Tensor:
        return self.features_extractor(obs)


class DrQPolicy(SACPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Box,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            use_sde: bool = False,
            log_std_init: float = -3,
            use_expln: bool = False,
            clip_mean: float = 2.0,
            optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            encoder_optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 2,
            encoder_feature_dim: int = 256,
            img_pad: int = 4,
            tactile_pad: int = 4,
    ):

        sample_obs = observation_space.sample()
        assert isinstance(sample_obs, Dict), "observations must be a dict"
        obs_space_keys = sample_obs.keys()
        self.has_state = False
        self.state_dim = -1
        if 'state' in obs_space_keys:
            self.state_dim = sample_obs['state'].shape[-1]
            self.has_state = True

        self.img_pad = img_pad
        self.tactile_pad = tactile_pad
        self.encoder_feature_dim = encoder_feature_dim

        if encoder_optimizer_kwargs is None:
            encoder_optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if encoder_optimizer_kwargs == torch.optim.Adam:
                encoder_optimizer_kwargs["eps"] = 1e-5
        self.encoder_optimizer_kwargs = encoder_optimizer_kwargs
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            use_expln=use_expln,
            clip_mean=clip_mean,
            features_extractor_class=Trunk,
            features_extractor_kwargs=None,
            normalize_images=False,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=False,
        )

    def _build_encoder(self, lr_schedule: Schedule):
        sample_obs = self.observation_space.sample()
        encoder_feature_dim = self.encoder_feature_dim
        assert isinstance(sample_obs, Dict), "observations must be a dict"
        obs_space_keys = sample_obs.keys()
        # make all relevant encoders, for each encoder make a target
        has_image = has_tactile = False
        img_repr_dim = tactile_repr_dim = -1
        encoder_params = []
        if 'image' in obs_space_keys:
            self.image_key = 'image'
            obs_shape = sample_obs['image'].shape
            # build encoder and target encoder
            self.img_encoder = Encoder(obs_shape).to(self.device)
            has_image = True
            img_repr_dim = self.img_encoder.repr_dim
            encoder_params.extend(list(self.img_encoder.parameters()))
            self.img_aug = RandomShiftsAug(pad=self.img_pad)
        elif 'pixels' in obs_space_keys:
            self.image_key = 'pixels'
            # build encoder and target encoder
            obs_shape = sample_obs['pixels'].shape
            self.img_encoder = Encoder(obs_shape).to(self.device)
            has_image = True
            img_repr_dim = self.img_encoder.repr_dim
            encoder_params.extend(list(self.img_encoder.parameters()))
            self.img_aug = RandomShiftsAug(pad=self.img_pad)
        else:
            self.img_encoder = None
            self.image_key = None

        if 'tactile' in obs_space_keys:
            # build encoder and target encoder
            obs_shape = sample_obs['tactile'].shape
            self.tactile_encoder = Encoder(obs_shape).to(self.device)
            has_tactile = True
            tactile_repr_dim = self.tactile_encoder.repr_dim
            encoder_params.extend(list(self.tactile_encoder.parameters()))
            self.tactile_aug = RandomShiftsAug(pad=self.tactile_pad)
        else:
            self.tactile_encoder = None

        if has_image or has_tactile:
            self.features_extractor_kwargs = {
                'has_tactile': has_tactile,
                'has_image': has_image,
                'img_repr_dim': img_repr_dim,
                'tactile_repr_dim': tactile_repr_dim,
                'feature_dim': encoder_feature_dim,
                'has_state': self.has_state,
                'state_dim': self.state_dim,
            }
            self.encoder_optimizer = self.optimizer_class(
                encoder_params,
                lr=lr_schedule(1),  # type: ignore[call-arg]
                **self.encoder_optimizer_kwargs,
            )
        else:
            raise NotImplementedError

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_encoder(lr_schedule)
        super()._build(lr_schedule)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return DrQActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return DrQCritic(**critic_kwargs).to(self.device)

    def set_training_mode(self, mode: bool) -> None:
        super().set_training_mode(mode)
        if self.img_encoder is not None:
            self.img_encoder.train(mode)
        if self.tactile_encoder is not None:
            self.tactile_encoder.train(mode)

    def encode_observation(self, obs: TensorDict, detach: bool = False, augment: bool = False):
        # if we have both tactile and image observations
        if self.img_encoder is not None and self.tactile_encoder is not None:
            # extract tactile and image embedding
            img_obs = obs[self.image_key].float()
            tactile_obs = obs['tactile'].float()
            if augment:
                img_obs = self.img_aug(img_obs)
                tactile_obs = self.tactile_aug(tactile_obs)
            obs_embed = self.img_encoder(img_obs)
            tact_embd = self.tactile_encoder(tactile_obs)
            # merge the two to extract final embedding
            embed = torch.cat([obs_embed, tact_embd], dim=-1)
        elif self.img_encoder is not None:
            img_obs = obs[self.image_key].float()
            if augment:
                img_obs = self.img_aug(img_obs)
            embed = self.img_encoder(img_obs)
        elif self.tactile_encoder is not None:
            tactile_obs = obs['tactile'].float()
            if augment:
                tactile_obs = self.tactile_aug(tactile_obs)
            embed = self.tactile_encoder(tactile_obs)
        else:
            raise NotImplementedError
        if 'state' in obs.keys():
            # add state the end of embedding
            z = obs['state']
            embed = torch.cat([embed, z], dim=-1)
        if detach:
            return embed.detach()
        else:
            return embed

    def forward(self, obs: PyTorchObs, deterministic: bool = False, detach: bool = True) -> torch.Tensor:
        return self._predict(obs, deterministic=deterministic, detach=detach)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False, detach: bool = True) -> torch.Tensor:
        # policy always use current encoder, i.e., not target, for picking the action
        observation = self.encode_observation(observation, detach=detach)
        return self.actor(observation, deterministic)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            encoder_feature_dim=self.encoder_feature_dim,
        )
        return data


class DrQ(SAC):
    def __init__(self,
                 learning_rate_encoder: Union[float, Schedule] = 1e-3,
                 policy: Type[DrQPolicy] = DrQPolicy,
                 *args,
                 **kwargs,
                 ):
        self.learning_rate_encoder = learning_rate_encoder
        self.log_image = True
        super().__init__(policy=policy, *args, **kwargs)
        assert isinstance(self.replay_buffer, DictReplayBuffer)
        assert isinstance(self.policy, DrQPolicy)

    def encode_observation(self, observation: TensorDict, detach: bool = True, augment: bool = False):
        return self.policy.encode_observation(observation, detach=detach, augment=augment)

    def _setup_lr_schedule(self) -> None:
        """Transform to callable if needed."""
        self.lr_schedule = get_schedule_fn(self.learning_rate)
        self.lr_schedule_encoder = get_schedule_fn(self.learning_rate_encoder)

    def _update_encoder_learning_rate(self) -> None:
        self.logger.record("train/encoder_learning_rate", self.lr_schedule_encoder(self._current_progress_remaining))
        update_learning_rate(self.policy.encoder_optimizer,
                             self.lr_schedule_encoder(self._current_progress_remaining))

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            tb_log_name: str = "DrQ",
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

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)
        self._update_encoder_learning_rate()

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            # get encoded state and obs after augmentation
            obs = self.encode_observation(observation=replay_data.observations, detach=False, augment=True)
            next_obs = self.encode_observation(observation=replay_data.next_observations, detach=True, augment=True)
            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            # detach latent state
            actions_pi, log_prob = self.actor.action_log_prob(obs.detach())
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
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
                # get next latent state from target encoder
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(next_obs)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(next_obs, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network using the latent state
            # using action from the replay buffer
            current_q_values = self.critic(obs, replay_data.actions)  # gradient of encoder flows through the critic

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.policy.encoder_optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            self.policy.encoder_optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(obs.detach(), actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
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

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


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

    algorithm = DrQ(
        env=vec_env,
        seed=0,
        buffer_size=100_000,
        **algorithm_kwargs,
    )

    algorithm.learn(
        total_timesteps=1000,
        log_interval=1,
    )
