from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from gymnasium.wrappers.time_limit import TimeLimit
from multimexmf.commons.redq import REDQ
from multimexmf.commons.max_entropy_algorithms.max_entropy_redq import MaxEntropyREDQ
from multimexmf.commons.max_entropy_algorithms.max_entropy_sac import MaxEntropySAC
from stable_baselines3 import SAC
import wandb
from wandb.integration.sb3 import WandbCallback
import numpy as np
import os
import sys
import argparse
from experiments.utils import Logger, hash_dict


def experiment(
        alg='SAC',
        logs_dir: str = './logs/',
        project_name: str = 'MCTest',
        total_steps: int = 25_000,
        num_envs: int = 8,
        normalize: bool = False,
        seed: int = 0,
):
    tb_dir = logs_dir + 'runs'

    config = dict(
        alg=alg,
        total_steps=total_steps,
        num_envs=num_envs,
        normalize=normalize,
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
    normalize = normalize

    vec_env = make_vec_env("Humanoid-v4", n_envs=num_envs, seed=seed, wrapper_class=TimeLimit,
                           env_kwargs={'render_mode': 'rgb_array'},
                           wrapper_kwargs={'max_episode_steps': 1_000})
    if normalize:
        vec_env = VecNormalize(venv=vec_env)

    algorithm_kwargs = {
        'policy': 'MlpPolicy',
        # 'train_freq': 32,
        # 'gradient_steps': 32,
        # 'learning_rate': 1e-3,
        'verbose': 1,
        'tensorboard_log': f"{tb_dir}/{run.id}",
        'gradient_steps': -1,
        'learning_starts': 5_000,
    }

    if alg == 'SAC':
        algorithm = SAC(
            env=vec_env,
            seed=seed,
            **algorithm_kwargs,
        )
    elif alg == 'REDQ':
        algorithm_kwargs.pop('policy')
        algorithm_kwargs.pop('gradient_steps')
        algorithm_kwargs['policy_kwargs'] = {'target_drop_rate': 0.1}
        algorithm = REDQ(
            env=vec_env,
            seed=seed,
            **algorithm_kwargs,
        )
    elif alg == 'MaxEntropySAC':
        ensemble_model_kwargs = {
            'learn_std': False,
            'optimizer_kwargs': {'lr': 3e-4, 'weight_decay': 0.0},
        }
        dynnamics_entropy_schedule = lambda x: float(x > 0.0)
        algorithm = MaxEntropySAC(
            env=vec_env,
            seed=seed,
            ensemble_model_kwargs=ensemble_model_kwargs,
            dynamics_entropy_schedule=dynnamics_entropy_schedule,
            **algorithm_kwargs,
        )
    elif alg == 'MaxEntropyREDQ':
        algorithm_kwargs.pop('policy')
        algorithm_kwargs.pop('gradient_steps')
        algorithm_kwargs['policy_kwargs'] = {'target_drop_rate': 0.1}
        ensemble_model_kwargs = {
            'learn_std': False,
            'optimizer_kwargs': {'lr': 3e-4, 'weight_decay': 0.0},
        }
        dynnamics_entropy_schedule = lambda x: float(x > 0.0)
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
        callback=[WandbCallback()],
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
        alg=args.alg,
        logs_dir=args.logs_dir,
        project_name=args.project_name,
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        normalize=bool(args.normalize),
        seed=args.seed,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTTest')

    # general experiment args
    parser.add_argument('--logs_dir', type=str, default='./logs/')
    parser.add_argument('--project_name', type=str, default='MCTest')
    parser.add_argument('--alg', type=str, default='MaxEntropySAC')
    parser.add_argument('--total_steps', type=int, default=1_000_000)
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--normalize', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_result_folder', type=str, default=None)

    args = parser.parse_args()
    main(args)
