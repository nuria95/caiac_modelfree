from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder
from stable_baselines3.sac import SAC
from gymnasium.wrappers.time_limit import TimeLimit
from multimexmf.commons.max_entropy_algorithms.max_entropy_sac import MaxEntropySAC
from multimexmf.commons.max_entropy_algorithms.max_entropy_redq import MaxEntropyREDQ
from multimexmf.commons.redq import REDQ
import wandb
from wandb.integration.sb3 import WandbCallback
import numpy as np
import os
import sys
import argparse
from experiments.utils import Logger, hash_dict
from gymnasium.envs.classic_control.pendulum import PendulumEnv
from typing import Optional


def experiment(
        alg: str = 'Disagreement',
        logs_dir: str = './logs/',
        project_name: str = 'MCTest',
        total_steps: int = 25_000,
        ensemble_type: str = 'MlpEns',
        num_envs: int = 8,
        normalize: bool = False,
        record_video: bool = False,
        ensemble_lr: float = 1e-3,
        ensemble_wd: float = 1e-4,
        entropy_switch: float = 0.5,
        seed: int = 0,
):
    tb_dir = logs_dir + 'runs'

    config = dict(
        alg=alg,
        total_steps=total_steps,
        num_envs=num_envs,
        normalize=normalize,
        record_video=record_video,
        ensemble_lr=ensemble_lr,
        ensemble_wd=ensemble_wd,
        exploitation_switch_at=entropy_switch,
        ensemble_type=ensemble_type,
    )

    # wandb.tensorboard.patch(root_logdir=tb_dir)
    run = wandb.init(
        dir=logs_dir,
        project=project_name,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        config=config,
    )
    record_video = record_video
    normalize = normalize

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

    vec_env = make_vec_env(CustomPendulumEnv, n_envs=num_envs, seed=seed, wrapper_class=TimeLimit,
                           env_kwargs={'render_mode': 'rgb_array'},
                           wrapper_kwargs={'max_episode_steps': 200})
    eval_env = make_vec_env(CustomPendulumEnv, n_envs=4, seed=seed + 1000,
                            env_kwargs={'render_mode': 'rgb_array'},
                            wrapper_class=TimeLimit,
                            wrapper_kwargs={'max_episode_steps': 200}
                            )
    if normalize:
        vec_env = VecNormalize(venv=vec_env)
        eval_env = VecNormalize(venv=eval_env)

    if record_video:
        callback = EvalCallback(VecVideoRecorder(eval_env,
                                                 video_folder=logs_dir + 'eval/',
                                                 record_video_trigger=lambda x: True,
                                                 ),
                                log_path=logs_dir,
                                best_model_save_path=logs_dir,
                                eval_freq=1000,
                                n_eval_episodes=5,
                                deterministic=True,
                                render=True)
    else:
        callback = EvalCallback(eval_env,
                                log_path=logs_dir,
                                best_model_save_path=logs_dir,
                                eval_freq=200,
                                n_eval_episodes=5,
                                deterministic=True,
                                render=False
                                )

    algorithm_kwargs = {
        'policy': 'MlpPolicy',
        # 'train_freq': 32,
        # 'gradient_steps': 32,
        # 'learning_rate': 1e-3,
        'verbose': 1,
        'tensorboard_log': f"{tb_dir}/{run.id}",
        'gradient_steps': -1,
        'learning_starts': 200 * num_envs,
    }

    ensemble_model_kwargs = {
        'learn_std': False,
        'optimizer_kwargs': {'lr': ensemble_lr, 'weight_decay': ensemble_wd},
    }

    if ensemble_type == 'MlpEns':
        from multimexmf.models.pretrain_models import EnsembleMLP
        ensemble_type = EnsembleMLP
    elif ensemble_type == 'MultiheadEns':
        from multimexmf.models.pretrain_models import MultiHeadGaussianEnsemble
        ensemble_type = MultiHeadGaussianEnsemble
    elif ensemble_type == 'DropoutEnsemble':
        from multimexmf.models.pretrain_models import DropoutEnsemble
        ensemble_type = DropoutEnsemble
    else:
        raise NotImplementedError

    if alg == 'SAC':
        algorithm = SAC(
            env=vec_env,
            seed=seed,
            **algorithm_kwargs,
        )
    elif alg == 'REDQ':
        algorithm_kwargs.pop('policy')
        algorithm_kwargs.pop('gradient_steps')
        algorithm = REDQ(
            env=vec_env,
            seed=seed,
            **algorithm_kwargs,
        )
    elif alg == 'MaxEntropySAC':
        dynnamics_entropy_schedule = lambda x: float(x > entropy_switch)
        algorithm = MaxEntropySAC(
            env=vec_env,
            seed=seed,
            ensemble_model_kwargs=ensemble_model_kwargs,
            dynamics_entropy_schedule=dynnamics_entropy_schedule,
            ensemble_type=ensemble_type,
            **algorithm_kwargs
        )
    elif alg == 'MaxEntropyREDQ':
        algorithm_kwargs.pop('policy')
        algorithm_kwargs.pop('gradient_steps')
        dynnamics_entropy_schedule = lambda x: float(x > entropy_switch)
        algorithm = MaxEntropyREDQ(
            env=vec_env,
            seed=seed,
            ensemble_model_kwargs=ensemble_model_kwargs,
            dynamics_entropy_schedule=dynnamics_entropy_schedule,
            **algorithm_kwargs,
        )
    else:
        raise NotImplementedError

    algorithm.learn(
        total_timesteps=total_steps,
        callback=[WandbCallback(), callback],
    )


def main(args):
    """"""
    from pprint import pprint
    print(args)
    """ generate experiment hash and set up redirect of output streams """
    exp_hash = hash_dict(args.__dict__)
    if args.exp_result_folder is not None:
        os.makedirs(args.exp_result_folder, exist_ok=True)
        log_file_path = os.path.join(args.exp_result_folder, '%s.log ' % exp_hash)
        logger = Logger(log_file_path)
        sys.stdout = logger
        sys.stderr = logger

    pprint(args.__dict__)
    print('\n ------------------------------------ \n')

    """ Experiment core """
    np.random.seed(args.seed)

    experiment(
        logs_dir=args.logs_dir,
        project_name=args.project_name,
        alg=args.alg,
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        normalize=bool(args.normalize),
        record_video=bool(args.record_video),
        ensemble_lr=args.ensemble_lr,
        ensemble_wd=args.ensemble_wd,
        entropy_switch=args.entropy_switch,
        ensemble_type=args.ensemble_type,
        seed=args.seed,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTTest')

    # general experiment args
    parser.add_argument('--logs_dir', type=str, default='./logs/')
    parser.add_argument('--project_name', type=str, default='MCTest')
    parser.add_argument('--alg', type=str, default='MaxEntropyREDQ')
    parser.add_argument('--total_steps', type=int, default=20_000)
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--normalize', type=int, default=0)
    parser.add_argument('--record_video', type=int, default=0)
    parser.add_argument('--ensemble_lr', type=float, default=3e-4)
    parser.add_argument('--ensemble_wd', type=float, default=0.0)
    parser.add_argument('--entropy_switch', type=float, default=0.5)
    parser.add_argument('--ensemble_type', type=str, default='MlpEns')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--exp_result_folder', type=str, default=None)

    args = parser.parse_args()
    main(args)
