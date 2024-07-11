from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder
from gymnasium.wrappers.time_limit import TimeLimit
from multimexmf.commons.intrinsic_reward_algorithms.sac_exploit_and_play import SacExploitAndPlay
from multimexmf.commons.intrinsic_reward_algorithms.utils import exploration_frequency_schedule, \
    DisagreementIntrinsicReward, CuriosityIntrinsicReward
import wandb
from wandb.integration.sb3 import WandbCallback
import numpy as np
import os
import sys
import argparse
from experiments.utils import Logger, hash_dict



def experiment(
        alg: str = 'Disagreement',
        logs_dir: str = './logs/',
        project_name: str = 'MCTest',
        total_steps: int = 25_000,
        num_envs: int = 8,
        normalize: bool = False,
        record_video: bool = False,
        ensemble_lr: float = 1e-3,
        ensemble_wd: float = 1e-4,
        exploitation_switch_at: float = 0.25,
        seed: int = 0,
):
    from multimexmf.envs.dm2gym import DMCGym
    tb_dir = logs_dir + 'runs'

    config = dict(
        alg=alg,
        total_steps=total_steps,
        num_envs=num_envs,
        normalize=normalize,
        record_video=record_video,
        ensemble_lr=ensemble_lr,
        ensemble_wd=ensemble_wd,
        exploitation_switch_at=exploitation_switch_at,
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

    env = lambda: TimeLimit(DMCGym(
        domain='reacher',
        task='hard',
        render_mode='rgb_array',
    ), max_episode_steps=50)

    vec_env = make_vec_env(env, n_envs=num_envs, seed=seed)
    eval_env = make_vec_env(env, n_envs=num_envs, seed=seed + 1_000)

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
                                eval_freq=1000,
                                n_eval_episodes=5,
                                deterministic=True,
                                render=False
                                )
    algorithm_kwargs = {
        'policy': 'MlpPolicy',
        # 'train_freq': 32,
        # 'gradient_steps': 32,
        'learning_rate': 1e-3,
        'verbose': 1,
        'tensorboard_log': f"{tb_dir}/{run.id}",
    }

    ensemble_model_kwargs = {
        'learn_std': False,
        'optimizer_kwargs': {'lr': ensemble_lr, 'weight_decay': ensemble_wd}
    }
    exploration_freq = [[1, 1], [exploitation_switch_at, 1_000_000]]
    if alg == 'Disagreement':
        intrinsic_reward_model = DisagreementIntrinsicReward
    elif alg == 'Curiosity':
        intrinsic_reward_model = CuriosityIntrinsicReward
    elif alg == 'Random':
        intrinsic_reward_model = None
    elif alg == 'SAC':
        intrinsic_reward_model = None
        exploration_freq = [[1, 1_000_000]]
    else:
        raise NotImplementedError

    algorithm = SacExploitAndPlay(
        env=vec_env,
        seed=seed,
        ensemble_model_kwargs=ensemble_model_kwargs,
        intrinsic_reward_model=intrinsic_reward_model,
        exploration_freq=exploration_frequency_schedule(
            exploration_freq
        ),
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
        alg=args.alg,
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        normalize=bool(args.normalize),
        record_video=bool(args.record_video),
        ensemble_lr=args.ensemble_lr,
        ensemble_wd=args.ensemble_wd,
        exploitation_switch_at=args.exploitation_switch_at,
        seed=args.seed,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTTest')

    # general experiment args
    parser.add_argument('--logs_dir', type=str, default='./logs/')
    parser.add_argument('--project_name', type=str, default='MCTest')
    parser.add_argument('--alg', type=str, default='SAC')
    parser.add_argument('--total_steps', type=int, default=75_000)
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--normalize', type=int, default=0)
    parser.add_argument('--record_video', type=int, default=1)
    parser.add_argument('--ensemble_lr', type=float, default=3e-4)
    parser.add_argument('--ensemble_wd', type=float, default=1e-4)
    parser.add_argument('--exploitation_switch_at', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--exp_result_folder', type=str, default=None)

    args = parser.parse_args()
    main(args)