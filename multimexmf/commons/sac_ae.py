import collections

from multimexmf.models.encoder_decoder_models import SACAEPolicy, preprocess_obs
from stable_baselines3.sac import SAC
from typing import Type, Union
import torch as th
from stable_baselines3.common.type_aliases import MaybeCallback, Schedule, TrainFreq
from stable_baselines3.common.utils import polyak_update, get_schedule_fn, update_learning_rate
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.logger import Image


class SACAE(SAC):
    def __init__(self,
                 encoder_tau: float = 0.02,
                 decoder_latent_loss_weight: float = 0.0,
                 train_decoder_with_next_obs: bool = True,
                 learning_rate_encoder: Union[float, Schedule] = 1e-3,
                 image_logging_freq: int = 1_000,
                 policy: Type[SACAEPolicy] = SACAEPolicy,
                 *args,
                 **kwargs,
                 ):
        self.learning_rate_encoder = learning_rate_encoder
        self.encoder_tau = encoder_tau
        self.decoder_latent_loss_weight = decoder_latent_loss_weight
        self.train_decoder_with_next_obs = train_decoder_with_next_obs
        self.image_logging_freq = image_logging_freq
        self.log_image = True
        super().__init__(policy=policy, *args, **kwargs)
        assert isinstance(self.replay_buffer, DictReplayBuffer)
        assert isinstance(self.policy, SACAEPolicy)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
        self.has_encoder = self.policy.has_encoder

    def encode_observation(self, observation: TensorDict, detach: bool = True, target: bool = False):
        return self.policy.encode_observation(observation, detach, target=target)

    def _setup_lr_schedule(self) -> None:
        """Transform to callable if needed."""
        self.lr_schedule = get_schedule_fn(self.learning_rate)
        self.lr_schedule_encoder = get_schedule_fn(self.learning_rate_encoder)

    def _update_encoder_learning_rate(self) -> None:
        self.logger.record("train/encoder_learning_rate", self.lr_schedule_encoder(self._current_progress_remaining))
        update_learning_rate(self.policy.encoder_optimizer,
                             self.lr_schedule_encoder(self._current_progress_remaining))

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)
        if self.has_encoder:
            self._update_encoder_learning_rate()

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        decoder_losses = collections.defaultdict(list)

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            # Get latent state to sample actions
            state = self.encode_observation(observation=replay_data.observations, detach=False, target=False)
            actions_pi, log_prob = self.actor.action_log_prob(state.detach())
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
                next_state = self.encode_observation(observation=replay_data.next_observations,
                                                     detach=True, target=True)
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(next_state)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(next_state, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network using the latent state
            # using action from the replay buffer
            current_q_values = self.critic(state, replay_data.actions)  # gradient of encoder flows through the critic

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            if self.has_encoder:
                self.policy.encoder_optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()
                self.policy.encoder_optimizer.step()
            else:
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(state.detach(), actions_pi), dim=1)
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
                if self.has_encoder:
                    self.policy.soft_update_encoder(self.encoder_tau)

            # decoder training
            if self.has_encoder:
                self.policy.encoder_optimizer.zero_grad()
                # Get latent embedding for the state -> add next_obs to the data as well
                state = self.encode_observation(observation=replay_data.observations, detach=False, target=False)
                # decode the state and evaluate reconstruction error for the AE
                decoded_state = self.policy.decode(state)
                decoder_loss = self.policy.reconstruction_loss(prediction=decoded_state,
                                                               target=replay_data.observations)
                stacked_losses = th.stack([val for val in decoder_loss.values()])
                recon_loss = stacked_losses.mean()
                if self.train_decoder_with_next_obs:
                    next_state = self.encode_observation(observation=replay_data.next_observations,
                                                         detach=False, target=False)
                    # decode the state and evaluate reconstruction error for the AE
                    decoded_next_state = self.policy.decode(next_state)
                    decoder_next_state_loss = self.policy.reconstruction_loss(prediction=decoded_next_state,
                                                                              target=replay_data.next_observations)
                    stacked_losses = th.stack([val for val in decoder_next_state_loss.values()])
                    recon_loss += stacked_losses.mean()


                latent_loss = (0.5 * state.pow(2).sum(1)).mean()
                total_loss = recon_loss + self.decoder_latent_loss_weight * latent_loss
                total_loss.backward()
                self.policy.encoder_optimizer.step()
                for key, val in decoder_loss.items():
                    decoder_losses[f"{key}_rc_loss"].append(val.item())
                decoder_losses['latent_loss'].append(latent_loss.item())
                if gradient_step == gradient_steps - 1 and self.log_image:
                    if self.policy.img_encoder is not None:
                        img_shape = replay_data.observations[self.policy.image_key][0].shape
                        n_frame = img_shape[0] // 3  # we use 3 channels
                        final_shape = (n_frame, 3) + img_shape[1:]
                        self.logger.record("train/true_image", Image(
                            replay_data.observations[self.policy.image_key][0].reshape(final_shape), "NCHW"),
                                           exclude=("stdout", "log", "json", "csv"))

                        reconstructed_image = decoded_state[self.policy.image_key][0].reshape(final_shape)
                        reconstructed_image = preprocess_obs(reconstructed_image, inverse=True)
                        self.logger.record("train/reconstructed_image", Image(reconstructed_image, "NCHW"),
                                           exclude=("stdout", "log", "json", "csv"))

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        if self.has_encoder:
            for key, val in decoder_losses.items():
                self.logger.record(f"train/{key}", np.mean(val))

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            tb_log_name: str = "SACAE",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ):
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None, "You must set the environment before calling learn()"
        assert isinstance(self.train_freq, TrainFreq)  # check done in _setup_learn()

        model_updates = 0
        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if not rollout.continue_training:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                self.log_image = model_updates % self.image_logging_freq == 0
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)
                model_updates += gradient_steps

        callback.on_training_end()

        return self


if __name__ == '__main__':
    from stable_baselines3.common.env_util import make_vec_env
    from gymnasium.wrappers.time_limit import TimeLimit
    from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
    from gymnasium.wrappers.pixel_observation import PixelObservationWrapper
    from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack




    env = lambda: PixelObservationWrapper(TimeLimit(HalfCheetahEnv(render_mode='rgb_array',
                                                                       ), max_episode_steps=1_000))
    print('using image observation')

    vec_env = VecFrameStack(make_vec_env(env, n_envs=4, seed=0), n_stack=3)
    eval_env = VecFrameStack(make_vec_env(env, n_envs=4, seed=1000), n_stack=3)

    algorithm_kwargs = {
        'learning_rate': 1e-3,
        'verbose': 1,
        'learning_starts': 1000,
        # 'tensorboard_log': "./logs/",
    }

    algorithm = SACAE(
        env=vec_env,
        seed=0,
        buffer_size=100_000,
        **algorithm_kwargs,
    )

    algorithm.learn(
        total_timesteps=1000,
        log_interval=1,
    )
