from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder
from gymnasium.wrappers.time_limit import TimeLimit
from multimexmf.commons.play_algorithm import \
    RandomOnPolicyPlayAlgorithm, CuriosityOnPolicyPlayAlgorithm, DisagreementOnPolicyPlayAlgorithm
import wandb
from wandb.integration.sb3 import WandbCallback
import numpy as np
import os
import sys
import argparse
from experiments.utils import Logger, hash_dict


def experiment(
        logs_dir: str = './logs/',
        project_name: str = 'MCTest',
        alg: str = 'SAC',
        total_steps: int = 25_000,
        num_envs: int = 8,
        num_steps: int = 256,
        normalize: bool = False,
        record_video: bool = False,
        intrinsic_reward_gradient_steps: int = 100,
        exploration_steps_per_exploitation_gradient_updates: int = 1,
        ensemble_lr: float = 1e-3,
        ensemble_wd: float = 1e-4,
        seed: int = 0,
):
    tb_dir = logs_dir + 'runs'

    config = dict(
        alg=alg,
        total_steps=total_steps,
        num_envs=num_envs,
        num_steps=num_steps,
        normalize=normalize,
        record_video=record_video,
        intrinsic_reward_gradient_steps=intrinsic_reward_gradient_steps,
        exploration_steps_per_exploitation_gradient_updates=exploration_steps_per_exploitation_gradient_updates,
        ensemble_lr=ensemble_lr,
        ensemble_wd=ensemble_wd,
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

    vec_env = make_vec_env('MountainCarContinuous-v0', n_envs=num_envs, seed=0, wrapper_class=TimeLimit,
                           env_kwargs={'render_mode': 'rgb_array'},
                           wrapper_kwargs={'max_episode_steps': 1_000})
    eval_env = make_vec_env('MountainCarContinuous-v0', n_envs=4, seed=1,
                            env_kwargs={'render_mode': 'rgb_array'},
                            wrapper_class=TimeLimit,
                            wrapper_kwargs={'max_episode_steps': 1_000}
                            )
    if normalize:
        vec_env = VecNormalize(venv=vec_env)
        eval_env = VecNormalize(venv=eval_env)

    if record_video:
        eval_callback = EvalCallback(VecVideoRecorder(eval_env,
                                                      video_folder=logs_dir + 'eval/',
                                                      record_video_trigger=lambda x: True,
                                                      ),
                                     log_path=logs_dir,
                                     best_model_save_path=logs_dir,
                                     eval_freq=num_steps,
                                     n_eval_episodes=5,
                                     deterministic=True,
                                     render=True)
    else:
        eval_callback = EvalCallback(eval_env,
                                     log_path=logs_dir,
                                     best_model_save_path=logs_dir,
                                     eval_freq=num_steps,
                                     n_eval_episodes=5,
                                     deterministic=True,
                                     render=False
                                     )

    base_algorithm_cls = PPO
    base_algorithm_kwargs = {
        'policy': 'MlpPolicy',
        'verbose': 1,
        'n_steps': num_steps,
        'gae_lambda': 0.9,
        'gamma': 0.9999,
        'n_epochs': 10,
        'learning_rate': 3e-4,
        'clip_range': 0.2,
        'ent_coef': 0.0,
        'use_sde': True,
        'sde_sample_freq': 4,
        'max_grad_norm': 5,
        'vf_coef': 0.19,
        'batch_size': num_envs * num_steps,
        'tensorboard_log': f"{tb_dir}/{run.id}"
    }

    exploitation_algorithm_cls = SAC
    exploitation_algorithm_kwargs = {
        'learning_rate': 3e-4,
        'buffer_size': 50000,
        'batch_size': 512,
        'ent_coef': 0.1,
        'train_freq': 1,
        'gradient_steps': 1,
        'tau': 0.01,
        'learning_starts': 0,
        'policy_kwargs': {'log_std_init': -3.67, 'net_arch': [64, 64]},
        'policy': 'MlpPolicy',
        'verbose': 1,
        'gamma': 0.9999,
        'tensorboard_log': f"{tb_dir}/{run.id}"
    }

    ensemble_model_kwargs = {
        'learn_std': False,
        'optimizer_kwargs': {'lr': ensemble_lr, 'weight_decay': ensemble_wd}
    }

    if alg == 'SAC':
        play_algorithm = RandomOnPolicyPlayAlgorithm(
            base_algorithm_cls=base_algorithm_cls,
            exploitation_algorithm_cls=exploitation_algorithm_cls,
            env=vec_env,
            base_algorithm_kwargs=base_algorithm_kwargs,
            exploitation_algorithm_kwargs=exploitation_algorithm_kwargs,
            ensemble_model_kwargs=ensemble_model_kwargs,
            intrinsic_reward_gradient_steps=intrinsic_reward_gradient_steps,
            exploration_steps_per_exploitation_gradient_updates=exploration_steps_per_exploitation_gradient_updates,
            exploitation_learning_starts=0,
            pred_diff=True,
            seed=seed,
        )
        total_exploration_timesteps = 0
        total_exploitation_timesteps = total_steps
    else:
        total_exploration_timesteps = (total_steps // (num_envs * num_steps)) * (num_envs * num_steps)
        total_exploitation_timesteps = 0
        if alg == 'Random':
            play_algorithm = RandomOnPolicyPlayAlgorithm(
                base_algorithm_cls=base_algorithm_cls,
                exploitation_algorithm_cls=exploitation_algorithm_cls,
                env=vec_env,
                base_algorithm_kwargs=base_algorithm_kwargs,
                exploitation_algorithm_kwargs=exploitation_algorithm_kwargs,
                ensemble_model_kwargs=ensemble_model_kwargs,
                intrinsic_reward_gradient_steps=intrinsic_reward_gradient_steps,
                exploration_steps_per_exploitation_gradient_updates=exploration_steps_per_exploitation_gradient_updates,
                exploitation_learning_starts=0,
                pred_diff=True,
                seed=seed,
            )

        elif alg == 'Curiosity':
            play_algorithm = CuriosityOnPolicyPlayAlgorithm(
                base_algorithm_cls=base_algorithm_cls,
                exploitation_algorithm_cls=exploitation_algorithm_cls,
                env=vec_env,
                base_algorithm_kwargs=base_algorithm_kwargs,
                exploitation_algorithm_kwargs=exploitation_algorithm_kwargs,
                ensemble_model_kwargs=ensemble_model_kwargs,
                intrinsic_reward_gradient_steps=intrinsic_reward_gradient_steps,
                exploration_steps_per_exploitation_gradient_updates=exploration_steps_per_exploitation_gradient_updates,
                exploitation_learning_starts=0,
                pred_diff=True,
                seed=seed,
            )

        elif alg == 'Disagreement':
            play_algorithm = DisagreementOnPolicyPlayAlgorithm(
                base_algorithm_cls=base_algorithm_cls,
                exploitation_algorithm_cls=exploitation_algorithm_cls,
                env=vec_env,
                base_algorithm_kwargs=base_algorithm_kwargs,
                exploitation_algorithm_kwargs=exploitation_algorithm_kwargs,
                ensemble_model_kwargs=ensemble_model_kwargs,
                intrinsic_reward_gradient_steps=intrinsic_reward_gradient_steps,
                exploration_steps_per_exploitation_gradient_updates=exploration_steps_per_exploitation_gradient_updates,
                exploitation_learning_starts=0,
                pred_diff=True,
                seed=seed,
            )

        else:
            raise NotImplementedError

    play_algorithm.learn(
        total_exploration_timesteps=total_exploration_timesteps,
        total_exploitation_timesteps=total_exploitation_timesteps,
        log_interval=1,
        base_callback=WandbCallback(),
        exploitation_callback=[WandbCallback(), eval_callback],
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
        num_steps=args.num_steps,
        normalize=bool(args.normalize),
        record_video=bool(args.record_video),
        intrinsic_reward_gradient_steps=args.intrinsic_reward_gradient_steps,
        exploration_steps_per_exploitation_gradient_updates=args.exploration_steps_per_exploitation_gradient_updates,
        ensemble_lr=args.ensemble_lr,
        ensemble_wd=args.ensemble_wd,
        seed=args.seed,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTTest')

    # general experiment args
    parser.add_argument('--logs_dir', type=str, default='./logs/')
    parser.add_argument('--project_name', type=str, default='MCTest')
    parser.add_argument('--alg', type=str, default='Disagreement')
    parser.add_argument('--total_steps', type=int, default=25_000)
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--num_steps', type=int, default=256)
    parser.add_argument('--normalize', type=int, default=0)
    parser.add_argument('--record_video', type=int, default=0)
    parser.add_argument('--intrinsic_reward_gradient_steps', type=int, default=100)
    parser.add_argument('--exploration_steps_per_exploitation_gradient_updates', type=int, default=1)
    parser.add_argument('--ensemble_lr', type=float, default=1e-3)
    parser.add_argument('--ensemble_wd', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--exp_result_folder', type=str, default=None)

    args = parser.parse_args()
    main(args)