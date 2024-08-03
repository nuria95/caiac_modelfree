import numpy as np
import torch
import torch as th
import torch.nn as nn
from torch.nn import functional as F
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.sac import SAC
from stable_baselines3.sac.policies import SACPolicy, get_action_dim, Actor, ContinuousCritic
from typing import Any, Dict, List, Optional, Type, Union
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule
from gymnasium import spaces
from stable_baselines3.common.utils import polyak_update
from multimexmf.models.pretrain_models import DropoutMlp as Mlp
from multimexmf.models.pretrain_models import dropout_weights_init_ as weights_init_


LOG_STD_MAX = 2
LOG_STD_MIN = -20
ACTION_BOUND_EPSILON = 1E-6


def get_probabilistic_num_min(num_mins):
    # allows the number of min to be a float
    floored_num_mins = np.floor(num_mins)
    if num_mins - floored_num_mins > 0.001:
        prob_for_higher_value = num_mins - floored_num_mins
        if np.random.uniform(0, 1) < prob_for_higher_value:
            return int(floored_num_mins + 1)
        else:
            return int(floored_num_mins)
    else:
        return num_mins


class REDQActor(Actor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # override the latent pi with
        # what is used in REDQ implementation from:
        # https://github.com/TakuyaHiraoka/Dropout-Q-Functions-for-Doubly-Efficient-Reinforcement-Learning
        if len(self.net_arch) > 1:
            self.latent_pi = Mlp(
                input_size=self.features_dim,
                output_size=self.net_arch[-1],
                hidden_sizes=self.net_arch[:-1],
                hidden_activation=self.activation_fn,
                # target_drop_rate=target_drop_rate, # not using layer norm and dropout for policy network
                # layer_norm=layer_norm,
            )
        self.apply(weights_init_)


class REDQCritic(ContinuousCritic):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Box,
            net_arch: List[int],
            features_extractor: BaseFeaturesExtractor,
            features_dim: int,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
            n_critics: int = 2,
            share_features_extractor: bool = True,
            target_drop_rate: float = 0.005,
            layer_norm: bool = True,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=net_arch,
            features_extractor=features_extractor,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
            features_dim=features_dim,
        )

        # redefine Q networks according REDQ
        self.target_drop_rate = target_drop_rate
        self.layer_norm = layer_norm

        action_dim = get_action_dim(self.action_space)
        self.q_networks: List[nn.Module] = []
        for idx in range(self.n_critics):
            q_net = Mlp(
                input_size=features_dim + action_dim,
                output_size=1,
                hidden_sizes=net_arch,
                hidden_activation=activation_fn,
                target_drop_rate=target_drop_rate,
                layer_norm=layer_norm)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)


class REDQPolicy(SACPolicy):
    def __init__(self,
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
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 n_critics: int = 2,
                 share_features_extractor: bool = False,
                 target_drop_rate: float = 0.005):
        self.target_drop_rate = target_drop_rate
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
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
        )

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> REDQActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return REDQActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> REDQCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return REDQCritic(target_drop_rate=self.target_drop_rate, **critic_kwargs).to(self.device)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        # setting training mode for the target critic to allow dropout
        self.critic_target.set_training_mode(mode)
        self.training = mode


class REDQ(SAC):
    def __init__(self,
                 policy: REDQPolicy = REDQPolicy,
                 gradient_steps: int = 20,
                 num_min_critics: int = 2,
                 q_target_mode: str = 'min',
                 policy_update_delay: int = -1,
                 *args,
                 **kwargs
                 ):
        self.num_min_critics = num_min_critics
        self.q_target_mode = q_target_mode
        if policy_update_delay > 0:
            self.policy_update_delay = policy_update_delay
        else:
            self.policy_update_delay = gradient_steps
        super().__init__(policy=policy, gradient_steps=gradient_steps, *args, **kwargs)
        assert isinstance(self.policy, REDQPolicy)
        self.n_critics = self.critic.n_critics

    def get_redq_q_target_no_grad(self, next_observations, rewards, done, ent_coef):
        # compute REDQ Q target, depending on the agent's Q target mode
        # allow min as a float:
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.action_log_prob(next_observations)
            next_log_prob = next_log_prob.reshape(-1, 1)
            if self.q_target_mode == 'min':
                num_mins_to_use = get_probabilistic_num_min(self.num_min_critics)
                sample_idxs = np.random.choice(self.n_critics, num_mins_to_use, replace=False)
                """Q target is min of a subset of Q values"""

                q_prediction_next_cat = torch.cat(self.critic_target(next_observations,
                                                                      next_actions), 1)[..., sample_idxs]
                min_q, _ = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
                next_q_with_log_prob = min_q - ent_coef * next_log_prob
                y_q = rewards + self.gamma * (1 - done) * next_q_with_log_prob
            elif self.q_target_mode == 'ave':
                """Q target is average of all Q values"""
                q_values = th.cat(self.critic_target(
                    next_observations, next_actions), dim=1).mean(dim=1).reshape(-1, 1)
                next_q_with_log_prob = q_values - ent_coef * next_log_prob
                y_q = rewards + self.gamma * (1 - done) * next_q_with_log_prob
            else:
                raise NotImplementedError
        return y_q

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # this function is called after each datapoint collected.
        # when we only have very limited data, we don't make updates
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

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

            """policy and alpha loss"""
            if ((gradient_step + 1) % self.policy_update_delay == 0) or gradient_step == gradient_steps - 1:
                # get policy loss
                actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
                log_prob = log_prob.reshape(-1, 1)
                q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
                qf_pi = th.mean(q_values_pi, dim=1, keepdim=True)
                actor_loss = (ent_coef * log_prob - qf_pi).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                    # Important: detach the variable from the graph
                    # so we don't change it with other losses
                    # see https://github.com/rail-berkeley/softlearning/issues/60
                    ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                    ent_coef_losses.append(ent_coef_loss.item())
                    self.ent_coef_optimizer.zero_grad()
                    ent_coef_loss.backward()
                    self.ent_coef_optimizer.step()

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

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DrO",
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
    from gymnasium.envs.classic_control.pendulum import PendulumEnv
    from gymnasium.wrappers.time_limit import TimeLimit
    from typing import Optional


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
    n_envs = 1
    vec_env = make_vec_env(CustomPendulumEnv, n_envs=4, seed=0, wrapper_class=TimeLimit,
                           env_kwargs={'render_mode': 'rgb_array'},
                           wrapper_kwargs={'max_episode_steps': 200})
    algorithm_kwargs = {
        # 'train_freq': 32,
        # 'gradient_steps': 32,
        'verbose': 1,
    }

    algorithm = REDQ(
        env=vec_env,
        **algorithm_kwargs
    )
    algorithm.learn(
        total_timesteps=30_000,
    )