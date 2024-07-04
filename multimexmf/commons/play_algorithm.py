from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from typing import Union, Type, TypeVar
from gymnasium import spaces
import torch as th
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import obs_as_tensor
import numpy as np
from multimexmf.models.pretrain_models import EnsembleMLP

SelfPlayAlgorithm = TypeVar("SelfPlayAlgorithm", bound="PlayAlgorithm")
from abc import ABC, abstractmethod


class BasePlayAlgorithm(ABC):
    def __init__(self,
                 base_algorithm_cls: Type[Union[OnPolicyAlgorithm, OffPolicyAlgorithm]],
                 exploitation_algorithm_cls: Type[OffPolicyAlgorithm],
                 env: Union[GymEnv, str, None],
                 base_algorithm_kwargs: dict,
                 exploitation_algorithm_kwargs: dict,
                 disagreement_model_kwargs: dict,
                 exploitation_learning_starts: int = 1000,
                 intrinsic_reward_gradient_steps: int = 1000,
                 intrinsic_reward_batch_size: int = 64,
                 ):
        # base algorithm processes env
        self.base_algorithm = base_algorithm_cls(env=env, **base_algorithm_kwargs)
        # Processed env is passed to the exploitation algorithm. Both envs refer to the same object
        self.exploitation_algorithm = exploitation_algorithm_cls(env=self.base_algorithm.env,
                                                                 **exploitation_algorithm_kwargs)
        self._setup_model(disagreement_model_kwargs)
        self.exploitation_learning_starts = exploitation_learning_starts
        self.intrinsic_reward_gradient_steps = intrinsic_reward_gradient_steps
        self.intrinsic_reward_batch_size = intrinsic_reward_batch_size

    @staticmethod
    def align_agents(current_agent: BaseAlgorithm, target_agent: BaseAlgorithm):
        # done to align model internal states
        target_agent._last_obs = current_agent._last_obs
        target_agent._last_original_obs = current_agent._last_original_obs
        target_agent._last_episode_starts = current_agent._last_episode_starts

    def _setup_model(self, disagreement_model_kwargs) -> None:
        obs_dim, act_dim = self.base_algorithm.observation_space.shape[0], self.base_algorithm.action_space.shape[0]
        output_dict = {'next_obs': self.base_algorithm.observation_space.sample()}
        self.ensemble_model = EnsembleMLP(
            input_dim=obs_dim + act_dim,
            output_dict=output_dict,
            **disagreement_model_kwargs,
        )

    def train(self, exploitation_agent_steps: int) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def get_intrinsic_reward(self, obs: th.Tensor, action: th.Tensor) -> th.Tensor:
        raise NotImplementedError

    @abstractmethod
    def learn(
            self,
            total_exploration_timesteps: int,
            total_exploitation_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 100,
            tb_log_name: str = "run",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ):
        """
        Return a trained model.

        :param total_exploration_timesteps: The total number of exploration samples (env steps) to train on
        :param total_exploitation_timesteps: The total number of exploitation samples (env steps) to train on
        :param callback: callback(s) called at every step with state of the algorithm.
        :param log_interval: for on-policy algos (e.g., PPO, A2C, ...) this is the number of
            training iterations (i.e., log_interval * n_steps * n_envs timesteps) before logging;
            for off-policy algos (e.g., TD3, SAC, ...) this is the number of episodes before
            logging.
        :param tb_log_name: the name of the run for TensorBoard logging
        :param reset_num_timesteps: whether or not to reset the current timestep number (used in logging)
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: the trained model
        """


class OnPolicyPlayAlgorithm(BasePlayAlgorithm):
    def __init__(self,
                 base_algorithm_cls: Type[OnPolicyAlgorithm],
                 exploitation_algorithm_cls: Type[OffPolicyAlgorithm],
                 env: Union[GymEnv, str, None],
                 base_algorithm_kwargs: dict,
                 exploitation_algorithm_kwargs: dict,
                 disagreement_model_kwargs: dict,
                 pred_diff: bool = True,
                 exploration_steps_per_exploitation_gradient_updates: int = 2,
                 exploitation_learning_starts: int = 1000,
                 intrinsic_reward_gradient_steps: int = 1000,
                 intrinsic_reward_batch_size: int = 64,
                 ):
        super().__init__(
            base_algorithm_cls=base_algorithm_cls,
            exploitation_algorithm_cls=exploitation_algorithm_cls,
            env=env,
            base_algorithm_kwargs=base_algorithm_kwargs,
            exploitation_algorithm_kwargs=exploitation_algorithm_kwargs,
            disagreement_model_kwargs=disagreement_model_kwargs,
            exploitation_learning_starts=exploitation_learning_starts,
            intrinsic_reward_gradient_steps=intrinsic_reward_gradient_steps,
            intrinsic_reward_batch_size=intrinsic_reward_batch_size,
        )
        self.pred_diff = pred_diff
        self.exploration_steps_per_exploitation_gradient_updates = exploration_steps_per_exploitation_gradient_updates

    def collect_rollouts(self,
                         env,
                         base_callback,
                         exploitation_callback,
                         rollout_buffer,
                         replay_buffer,
                         n_rollout_steps,
                         ):
        assert self.base_algorithm._last_obs is not None, "No previous observation was provided"
        assert n_rollout_steps > 0, "need to collect atleast 1 transition"
        # Switch to eval mode (this affects batch norm / dropout)
        self.base_algorithm.policy.set_training_mode(False)

        n_steps = 0
        # reset rollout buffer for on policy rollout
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.base_algorithm.use_sde:
            self.base_algorithm.policy.reset_noise(env.num_envs)

        base_callback.on_rollout_start()
        # TODO: Do we need to call exploitation callback on rollout start?
        exploitation_callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.base_algorithm.use_sde and self.base_algorithm.sde_sample_freq > 0 \
                    and n_steps % self.base_algorithm.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.base_algorithm.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self.base_algorithm._last_obs, self.base_algorithm.device)
                actions, values, log_probs = self.base_algorithm.policy(obs_tensor)
                # Get intrinsic reward for the exploration_agent
                intrinsic_rewards = self.get_intrinsic_reward(obs_tensor, actions)
            actions = actions.cpu().numpy()
            intrinsic_rewards = intrinsic_rewards.cpu().numpy().reshape(-1)

            # Rescale and perform action
            clipped_actions = actions
            buffer_action = actions
            if isinstance(self.base_algorithm.action_space, spaces.Box):
                if self.base_algorithm.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.base_algorithm.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.base_algorithm.action_space.low,
                                              self.base_algorithm.action_space.high)
                    if self.exploitation_algorithm.policy.squash_output:
                        buffer_action = self.base_algorithm.policy.scale_action(clipped_actions)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.base_algorithm.num_timesteps += env.num_envs

            # Give access to local variables
            base_callback.update_locals(locals())
            # TODO: Is this really necessary?
            exploitation_callback.update_locals(locals())
            if not base_callback.on_step():
                return False
            if not exploitation_callback.on_step():
                return False

            # Store data in replay buffer (normalized action and unnormalized observation)
            # store true reward in the buffer
            # automatically updates the internal state of the algorithm: last_obs =  next_obs
            self.exploitation_algorithm._store_transition(replay_buffer, buffer_action, new_obs, rewards, dones,
                                                          infos)  # type: ignore[arg-type]

            # TODO: SEE if this should be called
            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self.exploitation_algorithm._on_step()

            self.base_algorithm._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.base_algorithm.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.base_algorithm.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.base_algorithm.policy.predict_values(terminal_obs)[
                            0]  # type: ignore[arg-type]
                    intrinsic_rewards[idx] += self.base_algorithm.gamma * terminal_value
                    # rewards[idx] += self.base_algorithm.gamma * terminal_value
            rollout_buffer.add(
                self.base_algorithm._last_obs,  # type: ignore[arg-type]
                actions,
                intrinsic_rewards,
                self.base_algorithm._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            # update base algorithm internal state
            self.base_algorithm._last_obs = new_obs  # type: ignore[assignment]
            self.base_algorithm._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.base_algorithm.policy.predict_values(
                obs_as_tensor(new_obs, self.base_algorithm.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        base_callback.update_locals(locals())
        exploitation_callback.update_locals(locals())

        base_callback.on_rollout_end()
        exploitation_callback.on_rollout_end()

        return True

    def get_intrinsic_reward(self, obs: th.Tensor, action: th.Tensor) -> th.Tensor:
        self.ensemble_model.eval()
        features = self.extract_features(obs)
        inp = th.cat([features, action], dim=-1)
        predictions = self.ensemble_model(inp)
        disg = self.ensemble_model.get_disagreement(predictions)
        return disg

    def train(self, exploitation_agent_steps: int) -> None:
        self.base_algorithm.train()
        if self.base_algorithm.num_timesteps > 0 and \
                self.base_algorithm.num_timesteps > self.exploitation_learning_starts:
            # If no `gradient_steps` is specified,
            # do as many gradients steps as steps performed during the rollout
            # Special case when the user passes `gradient_steps=0`
            self.exploitation_algorithm.train(
                batch_size=self.exploitation_algorithm.batch_size,
                gradient_steps=exploitation_agent_steps)
        self.ensemble_model.train()
        for step in range(self.intrinsic_reward_gradient_steps):
            replay_data = self.exploitation_algorithm.replay_buffer.sample(
                self.intrinsic_reward_batch_size, env=self.exploitation_algorithm._vec_normalize_env)
            if self.pred_diff:
                target = {'next_obs': replay_data.next_observations - replay_data.observations}
            else:
                target = {'next_obs': replay_data.next_observations}

            features = self.extract_features(replay_data.observations)
            inp = th.cat([features, replay_data.actions], dim=-1)
            self.ensemble_model.optimizer.zero_grad()
            prediction = self.ensemble_model(inp)
            loss = self.ensemble_model.loss(prediction=prediction, target=target)
            total_loss = th.stack([val.reshape(-1, 1) for val in loss.values()], dim=-1).mean()
            total_loss.backward()

            self.ensemble_model.optimizer.step()
            # print('ensemble_loss: ', total_loss)

        self.ensemble_model.eval()

    def extract_features(self, obs):
        with th.no_grad():
            features = self.base_algorithm.policy.extract_features(
                obs, features_extractor=self.base_algorithm.policy.features_extractor)
        return features

    def learn(
            self,
            total_exploration_timesteps: int,
            total_exploitation_timesteps: int,
            base_callback: MaybeCallback = None,
            exploitation_callback: MaybeCallback = None,
            log_interval: int = 100,
            tb_log_name: str = "run",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ):
        iteration = 0
        initial_exploitation_callback = exploitation_callback
        # update callbacks for base and exploration algorithms. Setup learn, resets the environments and thus,
        # updates the internal states of the agents
        total_exploitation_timesteps, exploitation_callback = self.exploitation_algorithm._setup_learn(
            total_exploitation_timesteps,
            exploitation_callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        total_exploration_timesteps, base_callback = self.base_algorithm._setup_learn(
            total_exploration_timesteps,
            base_callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        # align internal env states after reset
        self.align_agents(current_agent=self.base_algorithm, target_agent=self.exploitation_algorithm)

        base_callback.on_training_start(locals(), globals())
        exploitation_callback.on_training_start(locals(), globals())

        assert self.base_algorithm.env is not None
        assert self.exploitation_algorithm.env is not None

        while self.base_algorithm.num_timesteps < total_exploration_timesteps:
            continue_training = self.collect_rollouts(
                env=self.base_algorithm.env,
                base_callback=base_callback,
                exploitation_callback=exploitation_callback,
                rollout_buffer=self.base_algorithm.rollout_buffer,
                replay_buffer=self.exploitation_algorithm.replay_buffer,
                n_rollout_steps=self.base_algorithm.n_steps
            )

            if not continue_training:
                break

            iteration += 1
            self.base_algorithm._update_current_progress_remaining(self.base_algorithm.num_timesteps,
                                                                   total_exploration_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.base_algorithm.ep_info_buffer is not None
                self.base_algorithm._dump_logs(iteration)

            exploitation_agent_steps = max((self.base_algorithm.n_steps * self.base_algorithm.n_envs) // \
                                           self.exploration_steps_per_exploitation_gradient_updates, 1)

            self.train(exploitation_agent_steps=exploitation_agent_steps)

        base_callback.on_training_end()

        if total_exploitation_timesteps > 0:
            print('Training exploitation policy')
            self.exploitation_algorithm.learn(
                total_timesteps=total_exploitation_timesteps,
                callback=initial_exploitation_callback,
                log_interval=log_interval,
                tb_log_name=tb_log_name,
                reset_num_timesteps=reset_num_timesteps,
                progress_bar=progress_bar,
            )
            # align internal states of the agents after training.
            self.align_agents(current_agent=self.exploitation_algorithm, target_agent=self.base_algorithm)

        return self


if __name__ == '__main__':
    from stable_baselines3 import SAC
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder
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


    log_dir = './logs2/'
    n_envs = 4
    vec_env = make_vec_env(CustomPendulumEnv, n_envs=4, seed=0, wrapper_class=TimeLimit,
                           env_kwargs={'render_mode': 'rgb_array'},
                           wrapper_kwargs={'max_episode_steps': 200})
    vec_env = VecVideoRecorder(venv=vec_env, video_folder=log_dir,
                               record_video_trigger=lambda x: True,
                               )

    n_steps = 1024
    eval_callback = EvalCallback(VecVideoRecorder(make_vec_env(CustomPendulumEnv, n_envs=4, seed=1,
                                                               env_kwargs={'render_mode': 'rgb_array'},
                                                               wrapper_class=TimeLimit,
                                                               wrapper_kwargs={'max_episode_steps': 200}
                                                               ),
                                                  video_folder=log_dir + 'eval/',
                                                  record_video_trigger=lambda x: True,
                                                  ),
                                 log_path=log_dir,
                                 best_model_save_path=log_dir,
                                 eval_freq=n_steps,
                                 n_eval_episodes=5, deterministic=True,
                                 render=True)
    base_algorithm_cls = PPO
    base_algorithm_kwargs = {
        'policy': 'MlpPolicy',
        'verbose': 1,
        'n_steps': n_steps,
        'gae_lambda': 0.95,
        'gamma': 0.9,
        'n_epochs': 10,
        'learning_rate': 1e-3,
        'clip_range': 0.2,
        'ent_coef': 0.0,
        'use_sde': True,
        'sde_sample_freq': 4,
    }

    exploitation_algorithm_cls = SAC
    exploitation_algorithm_kwargs = {
        'policy': 'MlpPolicy',
        'train_freq': 1,
        'gradient_steps': 2,
        'verbose': 1,
    }

    disagreement_model_kwargs = {
        'learn_std': False,
        'optimizer_kwargs': {'lr': 3e-4, 'weight_decay': 1e-4}
    }

    play_algorithm = OnPolicyPlayAlgorithm(
        base_algorithm_cls=base_algorithm_cls,
        exploitation_algorithm_cls=exploitation_algorithm_cls,
        env=vec_env,
        base_algorithm_kwargs=base_algorithm_kwargs,
        exploitation_algorithm_kwargs=exploitation_algorithm_kwargs,
        disagreement_model_kwargs=disagreement_model_kwargs,
        intrinsic_reward_gradient_steps=1000,
    )
    play_algorithm.learn(
        total_exploration_timesteps=25_000,
        total_exploitation_timesteps=0,
        exploitation_callback=eval_callback,
    )
