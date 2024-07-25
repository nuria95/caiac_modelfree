import numpy as np
import torch
import copy
import torch.nn as nn
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from multimexmf.commons.buffers.replay_buffers import NStepDictReplayBuffer
from stable_baselines3.ddpg import DDPG
from stable_baselines3.td3.policies import TD3Policy
from gymnasium import spaces
from typing import Optional, Type, Dict, Any, List, Union
from multimexmf.commons.drq import Trunk, Encoder, RandomShiftsAug, DrQCritic
from stable_baselines3.common.type_aliases import MaybeCallback, Schedule, GymEnv
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import polyak_update, get_schedule_fn, update_learning_rate
import torch.nn.functional as F
from multimexmf.models.encoder_decoder_models import weight_init
from stable_baselines3.td3.policies import Actor


class DrQv2Actor(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply(weight_init)

    def extract_features(self, obs: PyTorchObs, features_extractor: Trunk = None) -> torch.Tensor:
        return self.features_extractor(obs)


class DrQv2Policy(TD3Policy):
    def __init__(self,
                 observation_space: spaces.Space,
                 action_space: spaces.Box,
                 lr_schedule: Schedule,
                 net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
                 activation_fn: Type[nn.Module] = nn.ReLU,
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

    def make_actor(self, features_extractor: Optional[Trunk] = None) -> DrQv2Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return DrQv2Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[Trunk] = None) -> DrQCritic:
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
        return self.actor(observation)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            encoder_feature_dim=self.encoder_feature_dim,
        )
        return data


class LinearNormalActionNoise(ActionNoise):
    """
    A gaussian action noise with linear decay for the standard deviation.

    :param mean: (np.ndarray) the mean value of the noise
    :param sigma: (np.ndarray) the scale of the noise (std here)
    :param max_steps: (int)
    :param final_sigma: (np.ndarray)
    """

    def __init__(self, mean: np.ndarray, sigma: np.ndarray,
                 max_steps: Optional[int] = None, final_sigma: np.ndarray = None, sigma_clip: float = 0.3):
        self._mu = mean
        self._sigma = sigma
        self._step = 0
        self._max_steps = max_steps
        self.sigma_clip = sigma_clip
        if final_sigma is None:
            final_sigma = np.zeros_like(sigma)
        self._final_sigma = final_sigma

    def __call__(self):
        assert self.has_max_steps
        t = min(1.0, self._step / self._max_steps)
        sigma = (1 - t) * self._sigma + t * self._final_sigma
        self._step += 1
        return np.random.normal(self._mu, sigma)

    def get_current_sigma(self):
        t = min(1.0, max((self._step - 1), 0) / self._max_steps)
        sigma = (1 - t) * self._sigma + t * self._final_sigma
        return np.max(sigma), self.sigma_clip

    def set_max_steps(self, max_steps: int):
        self._max_steps = max_steps

    @property
    def has_max_steps(self):
        return self._max_steps is not None


class DrQv2(DDPG):
    def __init__(self,
                 env: Union[GymEnv, str],
                 learning_rate_encoder: Union[float, Schedule] = 1e-3,
                 policy: Type[DrQv2Policy] = DrQv2Policy,
                 replay_buffer_class: NStepDictReplayBuffer = NStepDictReplayBuffer,
                 action_noise: Optional[ActionNoise] = None,
                 n_steps: int = 3,
                 actor_tau: Optional[float] = None,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 *args,
                 **kwargs,
                 ):
        self.learning_rate_encoder = learning_rate_encoder
        self.log_image = True
        self.n_steps = n_steps
        if action_noise is None:
            # specify use constant sechedule if action noise is not specified
             sample = env.action_space.sample()
             action_noise = LinearNormalActionNoise(mean=np.zeros_like(sample),
                                                    sigma=np.ones_like(sample) * 0.2,
                                                    final_sigma=np.ones_like(sample) * 0.2,
                                                    sigma_clip=0.3,
                                                    max_steps=1,
                                                    )
        if policy_kwargs:
            if "n_critics" not in policy_kwargs:
                policy_kwargs["n_critics"] = 2
        else:
            policy_kwargs = {"n_critics": 2}
        super().__init__(policy=policy,
                         env=env,
                         replay_buffer_class=replay_buffer_class,
                         action_noise=action_noise,
                         policy_kwargs=policy_kwargs,
                         *args, **kwargs)
        assert isinstance(self.replay_buffer, NStepDictReplayBuffer)
        assert isinstance(self.policy, DrQv2Policy)
        self.replay_buffer.set_gamma(self.gamma)
        if actor_tau:
            self.actor_tau = actor_tau
        else:
            self.actor_tau = copy.deepcopy(self.tau)

        if isinstance(self.action_noise, LinearNormalActionNoise):
            # if max step is not specified, use the default for medium tough tasks in exploration.
            if not self.action_noise.has_max_steps:
                self.action_noise.set_max_steps(
                    max_steps=500_000 // self.n_envs
                )

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            tb_log_name: str = "DrQv2",
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

    @property
    def noise_std(self):
        if isinstance(self.action_noise, VectorizedActionNoise):
            if isinstance(self.action_noise.base_noise, LinearNormalActionNoise):
                noises = self.action_noise.noises
                return max([noise.get_current_sigma() for noise in noises])
        elif isinstance(self.action_noise, LinearNormalActionNoise):
            return self.action_noise.get_current_sigma()
        else:
            return self.target_policy_noise, self.target_noise_clip

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        self._update_encoder_learning_rate()

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env,
                                                    n_steps=self.n_steps)  # type: ignore[union-attr]

            obs = self.encode_observation(observation=replay_data.observations, detach=False, augment=True)
            with torch.no_grad():
                next_obs = self.encode_observation(observation=replay_data.next_observations, detach=True, augment=True)
                # Select action according to policy and add clipped noise
                noise_std, noise_clip = self.noise_std
                noise = replay_data.actions.clone().data.normal_(0, noise_std)
                noise = noise.clamp(-noise_clip, noise_clip)
                next_actions = (self.actor_target(next_obs) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = torch.cat(self.critic_target(next_obs, next_actions), dim=1)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(obs, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, torch.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.policy.encoder_optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            self.policy.encoder_optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                obs = obs.detach()
                noise = replay_data.actions.clone().data.normal_(0, noise_std)
                noise = noise.clamp(-noise_clip, noise_clip)
                actions = (self.actor(obs) + noise).clamp(-1, 1)
                q_values = torch.cat(self.critic(obs, actions), dim=1)
                q_values, _ = torch.min(q_values, dim=1, keepdim=True)
                actor_loss = - q_values.mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.actor_tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        exploitation_noise, _ = self.noise_std
        self.logger.record("train/exploitation_noise", exploitation_noise)


if __name__ == '__main__':
    from stable_baselines3.common.env_util import make_vec_env
    from gymnasium.wrappers.time_limit import TimeLimit
    from gymnasium.envs.mujoco.reacher_v4 import ReacherEnv
    from gymnasium.wrappers.pixel_observation import PixelObservationWrapper
    from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack

    env = lambda: PixelObservationWrapper(TimeLimit(ReacherEnv(render_mode='rgb_array',
                                                               height=84,
                                                               width=84), max_episode_steps=1_000))
    print('using image observation')

    vec_env = VecFrameStack(make_vec_env(env, n_envs=4, seed=0), n_stack=3)

    algorithm_kwargs = {
        'learning_rate': 1e-4,
        'verbose': 1,
        # 'learning_starts': 1000,
        # 'tensorboard_log': "./logs/",
    }

    algorithm = DrQv2(
        env=vec_env,
        seed=0,
        buffer_size=100_000,
        **algorithm_kwargs,
    )

    algorithm.learn(
        total_timesteps=1000,
        log_interval=1,
    )
