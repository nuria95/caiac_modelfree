from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder
from gymnasium.wrappers.time_limit import TimeLimit
from multimexmf.commons.drq import DrQ
from multimexmf.envs.action_repeat import ActionRepeat
import wandb
from wandb.integration.sb3 import WandbCallback
import numpy as np
import os
import sys
import argparse
from experiments.utils import Logger, hash_dict
from gymnasium.wrappers.pixel_observation import PixelObservationWrapper
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack


def experiment(
        logs_dir: str = './logs/',
        project_name: str = 'ReacherTest',
        total_steps: int = 25_000,
        num_envs: int = 8,
        record_video: bool = False,
        encoder_feature_dim: int = 50,
        seed: int = 0,
):
    from multimexmf.envs.dm2gym import DMCGym
    tb_dir = logs_dir + 'runs'

    config = dict(
        total_steps=total_steps,
        num_envs=num_envs,
        record_video=record_video,
        encoder_feature_dim=encoder_feature_dim,
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

    env = lambda: PixelObservationWrapper(TimeLimit(
        ActionRepeat(
            DMCGym(
                domain='reacher',
                task='easy',
                render_height=84,
                render_width=84,
            ), repeat=2),
        max_episode_steps=1_000))
    print('using image observation')

    vec_env = VecFrameStack(make_vec_env(env, n_envs=num_envs, seed=seed), n_stack=3)
    eval_env = VecFrameStack(make_vec_env(env, n_envs=4, seed=seed + 1000), n_stack=3)

    if record_video:
        callback = EvalCallback(VecVideoRecorder(eval_env,
                                                 video_folder=logs_dir + 'eval/',
                                                 record_video_trigger=lambda x: True,
                                                 ),
                                log_path=logs_dir,
                                best_model_save_path=logs_dir,
                                eval_freq=2_000,
                                n_eval_episodes=5,
                                deterministic=True,
                                render=True)
    else:
        callback = EvalCallback(eval_env,
                                log_path=logs_dir,
                                best_model_save_path=logs_dir,
                                eval_freq=2_000,
                                n_eval_episodes=5,
                                deterministic=True,
                                render=False
                                )

    buffer_size = min(total_steps, 400_000)
    print('buffer size', buffer_size)
    algorithm_kwargs = {
        'policy_kwargs': {
            'encoder_feature_dim': encoder_feature_dim,
            'net_arch': [1024, 1024]
        },
        'learning_rate': 1e-4,
        'learning_rate_encoder': 1e-4,
        'verbose': 1,
        'tensorboard_log': f"{tb_dir}/{run.id}",
        'buffer_size': buffer_size,
        'learning_starts': 1000,
        'batch_size': 256,
    }

    algorithm = DrQ(
        env=vec_env,
        seed=seed,
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
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        record_video=bool(args.record_video),
        encoder_feature_dim=args.encoder_feature_dim,
        seed=args.seed,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTTest')

    # general experiment args
    parser.add_argument('--logs_dir', type=str, default='./logs/')
    parser.add_argument('--project_name', type=str, default='ReacherTest')
    parser.add_argument('--total_steps', type=int, default=1_000_000)
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--record_video', type=int, default=0)
    parser.add_argument('--encoder_feature_dim', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_result_folder', type=str, default=None)

    args = parser.parse_args()
    main(args)
