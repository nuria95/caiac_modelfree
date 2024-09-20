from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers.time_limit import TimeLimit
import wandb
from wandb.integration.sb3 import WandbCallback
from experiments.max_rnd_sac.max_rnd_sac import MaxRNDSAC
import numpy as np
import os
import sys
import argparse
from experiments.utils import Logger, hash_dict


def experiment(
        domain_name: str = 'humanoid-walk',
        logs_dir: str = './logs/',
        project_name: str = 'MCTest',
        total_steps: int = 25_000,
        ensemble_type: str = 'MlpEns',
        num_envs: int = 1,
        ensemble_lr: float = 1e-3,
        ensemble_wd: float = 0.0,
        action_cost: float = 0.1,
        train_freq: int = 1,
        action_repeat: int = 1,
        seed: int = 0,
        features: int = 256,
):
    env_name, task = domain_name.split('-')
    from multimexmf.envs.dm2gym import DMCGym
    from multimexmf.envs.action_repeat import ActionRepeat
    from multimexmf.envs.action_cost import ActionCost
    tb_dir = logs_dir + 'runs'

    config = dict(
        alg='MaxRNDSAC',
        total_steps=total_steps,
        num_envs=num_envs,
        ensemble_lr=ensemble_lr,
        ensemble_wd=ensemble_wd,
        ensemble_type=ensemble_type,
        action_cost=action_cost,
        domain_name=domain_name,
        train_freq=train_freq,
        action_repeat=action_repeat,
        features=features,
    )

    # wandb.tensorboard.patch(root_logdir=tb_dir)
    run = wandb.init(
        dir=logs_dir,
        project=project_name,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        config=config,
        entity='sukhijab',
    )

    env = lambda: TimeLimit(
        ActionRepeat(
            ActionCost(DMCGym(
                domain=env_name,
                task=task,
                render_mode='rgb_array',
            ), action_cost=action_cost),
            repeat=action_repeat, return_total_reward=True),
        max_episode_steps=1_000)

    vec_env = make_vec_env(env, n_envs=num_envs, seed=seed)
    eval_env = make_vec_env(env, n_envs=num_envs, seed=seed + 1_000)

    callback = EvalCallback(eval_env,
                            log_path=logs_dir,
                            best_model_save_path=logs_dir,
                            eval_freq=1_000,
                            n_eval_episodes=5,
                            deterministic=True,
                            render=False
                            )

    algorithm_kwargs = {
        'policy': 'MlpPolicy',
        'train_freq': train_freq,
        # 'gradient_steps': 32,
        'verbose': 1,
        'tensorboard_log': f"{tb_dir}/{run.id}",
        'gradient_steps': -1,
        'learning_starts': 500 * num_envs,
    }

    ensemble_model_kwargs = {
        'learn_std': False,
        'features': (features, features),
        'optimizer_kwargs': {'lr': ensemble_lr, 'weight_decay': ensemble_wd},
    }


    algorithm = MaxRNDSAC(
        env=vec_env,
        seed=seed,
        ensemble_model_kwargs=ensemble_model_kwargs,
        **algorithm_kwargs
    )

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
        domain_name=args.domain_name,
        ensemble_type=args.ensemble_type,
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        ensemble_lr=args.ensemble_lr,
        ensemble_wd=args.ensemble_wd,
        train_freq=args.train_freq,
        action_repeat=args.action_repeat,
        action_cost=args.action_cost,
        seed=args.seed,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTTest')

    # general experiment args
    parser.add_argument('--logs_dir', type=str, default='./logs/')
    parser.add_argument('--project_name', type=str, default='MCTest')
    parser.add_argument('--domain_name', type=str, default='quadruped-run')
    parser.add_argument('--ensemble_type', type=str, default='MlpEns')
    parser.add_argument('--total_steps', type=int, default=250_000)
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--ensemble_lr', type=float, default=3e-4)
    parser.add_argument('--ensemble_wd', type=float, default=0.0)
    parser.add_argument('--train_freq', type=int, default=1)
    parser.add_argument('--action_repeat', type=int, default=1)
    parser.add_argument('--action_cost', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_result_folder', type=str, default=None)

    args = parser.parse_args()
    main(args)
