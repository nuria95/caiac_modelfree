import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.sac.policies import SACPolicy
from gymnasium import spaces
from gymnasium.spaces.box import Box
from typing import Any, Dict, List, Optional, Type, Union, Tuple
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
import copy
from stable_baselines3.common.preprocessing import is_image_space, maybe_transpose
from stable_baselines3.common.utils import is_vectorized_observation, obs_as_tensor
import math
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import polyak_update
import torch.nn.functional as F


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


# OUT_DIM = {2: 39, 4: 35, 6: 31}

# Most implementation is taken from: https://github.com/denisyarats/pytorch_sac_ae/tree/master


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2 ** bits
    if bits < 8:
        obs = torch.floor(obs / 2 ** (8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(self, dummy_obs, feature_dim, num_layers=2, num_filters=32):
        super().__init__()
        obs_shape = dummy_obs.shape
        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        self.outputs = dict()
        dummy_feat = self.forward_conv(torch.from_numpy(dummy_obs[None]))
        self.out_dim = dummy_feat.shape[-1]
        self.fc = nn.Linear(self.out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.reshape(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        out = torch.tanh(h_norm)
        self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class IdentityEncoder(nn.Module):
    def __init__(self, dummy_obs, feature_dim, num_layers, num_filters):
        super().__init__()
        obs_shape = dummy_obs.shape
        self.out_dim = obs_shape[-1]
        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder}


def make_encoder(
        encoder_type, dummy_obs, feature_dim, num_layers, num_filters
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        dummy_obs, feature_dim, num_layers, num_filters
    )


class PixelDecoder(nn.Module):
    def __init__(self, obs_shape, out_dim, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.out_dim = out_dim
        self.latent_pixel_size = int(math.sqrt(out_dim // num_filters))

        self.fc = nn.Linear(
            feature_dim, self.out_dim,
        )

        self.deconvs = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.deconvs.append(
                nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1)
            )
        self.deconvs.append(
            nn.ConvTranspose2d(
                num_filters, obs_shape[0], 3, stride=2, output_padding=1
            )
        )

        self.outputs = dict()

    def forward(self, h):
        h = torch.relu(self.fc(h))
        self.outputs['fc'] = h

        deconv = h.reshape(-1, self.num_filters, self.latent_pixel_size, self.latent_pixel_size)
        self.outputs['deconv1'] = deconv

        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))
            self.outputs['deconv%s' % (i + 1)] = deconv

        obs = self.deconvs[-1](deconv)
        self.outputs['obs'] = obs

        return obs

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_decoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_decoder/%s_i' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param(
                'train_decoder/deconv%s' % (i + 1), self.deconvs[i], step
            )
        L.log_param('train_decoder/fc', self.fc, step)


_AVAILABLE_DECODERS = {'pixel': PixelDecoder}


def make_decoder(
        decoder_type, obs_shape, out_dim, feature_dim, num_layers, num_filters
):
    assert decoder_type in _AVAILABLE_DECODERS
    return _AVAILABLE_DECODERS[decoder_type](
        obs_shape, out_dim, feature_dim, num_layers, num_filters
    )


class SACAEPolicy(SACPolicy):
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
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 2,
            share_features_extractor: bool = False,
            encoder_feature_dim: int = 256,
            num_encoder_layers: int = 2,
            num_encoder_filters: int = 32,
    ):

        sample_obs = observation_space.sample()
        assert isinstance(sample_obs, Dict), "observations must be a dict"
        obs_space_keys = sample_obs.keys()
        state_dim = 0
        if 'state' in obs_space_keys:
            state_dim += sample_obs['state'].shape[-1]
        # we add a additional features for the obs if we have pixels or tactile information
        if 'image' in obs_space_keys or 'pixels' in obs_space_keys or 'tactile' in obs_space_keys:
            state_dim += encoder_feature_dim

        assert state_dim > 0, "need to have at least state, image or tactile obs"

        self.state_dim = state_dim
        self.encoder_feature_dim = encoder_feature_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_encoder_filters = num_encoder_filters
        # define a feature space for the policy over which it operates
        feature_space = Box(
            low=np.ones(state_dim) * -np.inf,
            high=np.ones(state_dim) * np.inf,
        )
        self.full_observation_space = observation_space
        super().__init__(
            observation_space=feature_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            use_expln=use_expln,
            clip_mean=clip_mean,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
        )

    def _build_ae(self):
        sample_obs = self.full_observation_space.sample()
        encoder_feature_dim = self.encoder_feature_dim
        num_encoder_layers = self.num_encoder_layers
        num_encoder_filters = self.num_encoder_filters
        state_dim = self.state_dim
        assert isinstance(sample_obs, Dict), "observations must be a dict"
        obs_space_keys = sample_obs.keys()
        # make all relevant encoders, for each encoder make a target
        if 'image' in obs_space_keys:
            self.image_key = 'image'
            # build encoder and target encoder
            self.img_encoder = make_encoder(
                encoder_type='pixel',
                dummy_obs=sample_obs['image'],
                feature_dim=encoder_feature_dim,
                num_layers=num_encoder_layers,
                num_filters=num_encoder_filters,
            )
            self.target_img_encoder = make_encoder(
                encoder_type='pixel',
                dummy_obs=sample_obs['pixels'],
                feature_dim=encoder_feature_dim,
                num_layers=num_encoder_layers,
                num_filters=num_encoder_filters,
            )
            self.target_img_encoder.load_state_dict(self.img_encoder.state_dict())
            # build decoder
            self.img_decoder = make_decoder(
                decoder_type='pixel',
                obs_shape=sample_obs['image'].shape,
                out_dim=self.img_encoder.out_dim,
                feature_dim=state_dim,
                num_layers=num_encoder_layers,
                num_filters=num_encoder_filters,
            )

        elif 'pixels' in obs_space_keys:
            self.image_key = 'pixels'
            # build encoder and target encoder
            obs_shape = sample_obs['pixels'].shape
            self.img_encoder = make_encoder(
                encoder_type='pixel',
                dummy_obs=sample_obs['pixels'],
                feature_dim=encoder_feature_dim,
                num_layers=num_encoder_layers,
                num_filters=num_encoder_filters,
            )
            self.target_img_encoder = make_encoder(
                encoder_type='pixel',
                dummy_obs=sample_obs['pixels'],
                feature_dim=encoder_feature_dim,
                num_layers=num_encoder_layers,
                num_filters=num_encoder_filters,
            )
            self.target_img_encoder.load_state_dict(self.img_encoder.state_dict())
            # build decoder
            self.img_decoder = make_decoder(
                decoder_type='pixel',
                obs_shape=obs_shape,
                out_dim=self.img_encoder.out_dim,
                feature_dim=state_dim,
                num_layers=num_encoder_layers,
                num_filters=num_encoder_filters,
            )
        else:
            self.img_encoder = None
            self.image_key = None
            self.img_decoder = None

        if 'tactile' in obs_space_keys:
            # build encoder and target encoder
            obs_shape = sample_obs['tactile'].shape
            self.tactile_encoder = make_encoder(
                encoder_type='pixel',
                dummy_obs=sample_obs['tactile'],
                feature_dim=encoder_feature_dim,
                num_layers=num_encoder_layers,
                num_filters=num_encoder_filters,
            )
            self.target_tactile_encoder = make_encoder(
                encoder_type='pixel',
                dummy_obs=sample_obs['tactile'],
                feature_dim=encoder_feature_dim,
                num_layers=num_encoder_layers,
                num_filters=num_encoder_filters,
            )
            self.target_tactile_encoder.load_state_dict(self.tactile_encoder.state_dict())
            # build decoder
            self.tactile_decoder = make_decoder(
                decoder_type='pixel',
                obs_shape=obs_shape,
                out_dim=self.tactile_decoder.out_dim,
                feature_dim=state_dim,
                num_layers=num_encoder_layers,
                num_filters=num_encoder_filters,
            )
        else:
            self.tactile_encoder = None
            self.tactile_decoder = None

        if 'state' in obs_space_keys:
            state_dim += sample_obs['state'].shape[-1]

        self.img_tactile_module = None
        if self.img_encoder is not None and self.tactile_encoder is not None:
            # build module to merge tactile and image obs
            self.img_tactile_module = nn.Sequential(
                nn.Linear(encoder_feature_dim * 2, encoder_feature_dim * 2),
                nn.ReLU(),
                nn.Linear(encoder_feature_dim * 2, encoder_feature_dim)
            )
            self.target_img_tactile_module = nn.Sequential(
                nn.Linear(encoder_feature_dim * 2, encoder_feature_dim * 2),
                nn.ReLU(),
                nn.Linear(encoder_feature_dim * 2, encoder_feature_dim)
            )
            self.target_img_tactile_module.load_state_dict(self.img_tactile_module.state_dict())

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        self._build_ae()
        encoder_params = []

        if self.img_encoder is not None:
            self.img_encoder.to(self.device)
            self.target_img_encoder.to(self.device)
            encoder_params.extend(list(self.img_encoder.parameters()))
            self.img_decoder.to(self.device)
            encoder_params.extend(list(self.img_decoder.parameters()))

        if self.tactile_encoder is not None:
            self.tactile_encoder.to(self.device)
            self.target_tactile_encoder.to(self.device)
            encoder_params.extend(list(self.tactile_encoder.parameters()))
            self.tactile_decoder.to(self.device)
            encoder_params.extend(list(self.tactile_decoder.parameters()))

        if self.img_tactile_module is not None:
            self.img_tactile_module.to(self.device)
            self.target_img_tactile_module.to(self.device)
            encoder_params.extend(list(self.img_tactile_module.parameters()))

        if len(encoder_params) > 0:
            self.encoder_optimizer = self.optimizer_class(
                encoder_params,
                lr=lr_schedule(1),
                **self.optimizer_kwargs,
            )

    def set_training_mode(self, mode: bool) -> None:
        super().set_training_mode(mode)
        if self.img_encoder is not None:
            self.img_encoder.train(mode)
        if self.tactile_encoder is not None:
            self.tactile_encoder.train(mode)
        if self.img_tactile_module is not None:
            self.img_tactile_module.train(mode)

    def encode_observation(self, obs: TensorDict, detach: bool = False, target: bool = False):
        # if we have both tactile and image observations
        if self.img_encoder is not None and self.tactile_encoder is not None:
            if target:
                # extract tactile and image embedding
                obs_embed = self.target_img_encoder(obs[self.image_key], detach=detach)
                tact_embd = self.target_tactile_encoder(obs['tactile'], detach=detach)
                # merge the two to extract final embedding
                embed = torch.cat([obs_embed, tact_embd], dim=-1)
                embed = self.target_img_tactile_module(embed)
            else:
                # extract tactile and image embedding
                obs_embed = self.img_encoder(obs[self.image_key], detach=detach)
                tact_embd = self.tactile_encoder(obs['tactile'], detach=detach)
                # merge the two to extract final embedding
                embed = torch.cat([obs_embed, tact_embd], dim=-1)
                embed = self.img_tactile_module(embed)
        elif self.img_encoder is not None:
            if target:
                embed = self.target_img_encoder(obs[self.image_key], detach=detach)
            else:
                embed = self.img_encoder(obs[self.image_key], detach=detach)
        elif self.tactile_encoder is not None:
            if target:
                embed = self.target_tactile_encoder(obs['tactile'], detach=detach)
            else:
                embed = self.tactile_encoder(obs['tactile'], detach=detach)
        else:
            if 'state' not in obs.keys():
                raise Exception('No state, image or tactile obs found')
            return obs['state']
        if 'state' in obs.keys():
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
            num_encoder_layers=self.num_encoder_layers,
            num_encoder_filters=self.num_encoder_filters,
        )
        return data

    def decode(self, z: torch.Tensor) -> TensorDict:
        assert self.has_encoder, "need to have a decoder for decoding"
        obs = {}
        if self.img_decoder is not None:
            img = self.img_decoder(z)
            obs[self.image_key] = img
        if self.tactile_decoder is not None:
            tac = self.tactile_decoder(z)
            obs['tactile'] = tac
        return obs

    def reconstruction_loss(self, prediction: Dict, target: Dict):
        loss = {}
        for key, val in prediction.items():
            mean = target[key]
            # TODO: See if this is really needed -> we train with MSE and preprocess the image to be between -0.5, 0.5
            if key == self.image_key:
                mean = preprocess_obs(mean)
            loss[key] = F.mse_loss(mean, val)
        return loss

    @property
    def has_encoder(self):
        return self.img_encoder is not None or self.tactile_encoder is not None

    def obs_to_tensor(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[PyTorchObs, bool]:
        """adapted obs to tensor to use fulL_observation space and not the space of the embedding"""
        vectorized_env = False
        assert isinstance(observation, dict), "we only support dict observations for AE policy"
        if isinstance(observation, dict):
            assert isinstance(
                self.full_observation_space, spaces.Dict
            ), f"The observation provided is a dict but the obs space is {self.full_observation_space}"
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            observation = copy.deepcopy(observation)
            for key, obs in observation.items():
                obs_space = self.full_observation_space.spaces[key]
                if is_image_space(obs_space):
                    obs_ = maybe_transpose(obs, obs_space)
                else:
                    obs_ = np.array(obs)
                vectorized_env = vectorized_env or is_vectorized_observation(obs_, obs_space)
                # Add batch dimension if needed
                observation[key] = obs_.reshape((-1, *self.full_observation_space[key].shape))  # type: ignore[misc]

        obs_tensor = obs_as_tensor(observation, self.device)
        return obs_tensor, vectorized_env

    def soft_update_encoder(self, tau):
        if self.img_encoder is not None:
            polyak_update(self.img_encoder.parameters(), self.target_img_encoder.parameters(), tau)
        if self.tactile_encoder is not None:
            polyak_update(self.tactile_encoder.parameters(), self.target_tactile_encoder.parameters(), tau)
        if self.img_tactile_module is not None:
            polyak_update(self.img_tactile_module.parameters(), self.target_img_tactile_module.parameters(), tau)
