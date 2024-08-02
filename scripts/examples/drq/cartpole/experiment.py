from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder
from gymnasium.wrappers.time_limit import TimeLimit
from multimexmf.commons.drq import DrQ
from multimexmf.commons.drqv2 import DrQv2
from multimexmf.commons.intrinsic_reward_algorithms.drq_exploit_and_play import DrQExploitAndPlay
from multimexmf.commons.intrinsic_reward_algorithms.drqv2_exploit_and_play import DrQv2ExploitAndPlay, \
    LinearNormalActionNoise
from multimexmf.commons.intrinsic_reward_algorithms.utils import exploration_frequency_schedule, \
    DisagreementIntrinsicReward
from multimexmf.envs.action_repeat import ActionRepeat
from multimexmf.envs.action_cost import ActionCost
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
        project_name: str = 'CartPoleImgTest',
        alg: str = 'DrQ',
        total_steps: int = 25_000,
        num_envs: int = 8,
        record_video: bool = False,
        encoder_feature_dim: int = 50,
        task: str = 'swingup_sparse',
        exploitation_switch_at: float = 0.75,
        init_exploration_freq: int = 4,
        train_ensemble_with_target: bool = False,
        predict_img_embed: bool = False,
        update_encoder_with_exploration_policy: bool = False,
        seed: int = 0,
        action_cost: float = 0.0,
        sig: float = 0.1,
):
    from multimexmf.envs.dm2gym import DMCGym
    tb_dir = logs_dir + 'runs'

    config = dict(
        alg=alg,
        total_steps=total_steps,
        num_envs=num_envs,
        record_video=record_video,
        encoder_feature_dim=encoder_feature_dim,
        task=task,
        exploitation_switch_at=exploitation_switch_at,
        init_exploration_freq=init_exploration_freq,
        train_ensemble_with_target=train_ensemble_with_target,
        update_encoder_with_exploration_policy=update_encoder_with_exploration_policy,
        predict_img_embed=predict_img_embed,
        action_cost=action_cost,
        sig=sig,
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
            ActionCost(DMCGym(
                domain='cartpole',
                task=task,
                render_mode='rgb_array',
                render_height=84,
                render_width=84,
            ), action_cost=action_cost),
            repeat=2, return_total_reward=True),
        max_episode_steps=1_000))
    print('using image observation')

    vec_env = VecFrameStack(make_vec_env(env, n_envs=num_envs, seed=seed), n_stack=3)
    eval_env = VecFrameStack(make_vec_env(env, n_envs=1, seed=seed + 1000), n_stack=3)

    if record_video:
        callback = EvalCallback(VecVideoRecorder(eval_env,
                                                 video_folder=logs_dir + 'eval/',
                                                 record_video_trigger=lambda x: True,
                                                 ),
                                log_path=logs_dir,
                                best_model_save_path=logs_dir,
                                eval_freq=5_000,
                                n_eval_episodes=5,
                                deterministic=True,
                                render=True)
    else:
        callback = EvalCallback(eval_env,
                                log_path=logs_dir,
                                best_model_save_path=logs_dir,
                                eval_freq=5_000,
                                n_eval_episodes=5,
                                deterministic=True,
                                render=False
                                )

    buffer_size = 1_000_000
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
        'learning_starts': 4000,
        'batch_size': 256,
        'tau': 0.01,
        'gradient_steps': -1,
    }
    act = vec_env.action_space.sample()
    action_noise = LinearNormalActionNoise(
        mean=np.zeros_like(act),
        sigma=np.ones_like(action_cost) * 1.0,
        final_sigma=np.ones_like(action_cost) * sig,
        max_steps=500_000 // num_envs,
    )

    if alg == 'DrQ':
        algorithm = DrQ(
            env=vec_env,
            seed=seed,
            **algorithm_kwargs
        )

    elif alg == 'DisagreementDrQ':
        ensemble_model_kwargs = {
            'learn_std': False,
            'optimizer_kwargs': {'lr': 1e-4},
            'features': (1024, 1024),
        }
        exploration_freq = [[1, init_exploration_freq], [exploitation_switch_at, -1]]
        intrinsic_reward_model = DisagreementIntrinsicReward
        algorithm = DrQExploitAndPlay(
            env=vec_env,
            seed=seed,
            ensemble_model_kwargs=ensemble_model_kwargs,
            intrinsic_reward_model=intrinsic_reward_model,
            train_ensemble_with_target=train_ensemble_with_target,
            update_encoder_with_exploration_policy=update_encoder_with_exploration_policy,
            predict_img_embed=predict_img_embed,
            exploration_freq=exploration_frequency_schedule(
                exploration_freq
            ),
            **algorithm_kwargs
        )
    elif alg == 'DrQv2':
        algorithm = DrQv2(
            env=vec_env,
            seed=seed,
            actor_tau=1.0,
            action_noise=action_noise,
            **algorithm_kwargs
        )
    elif alg == 'DisagreementDrQv2':
        ensemble_model_kwargs = {
            'learn_std': False,
            'optimizer_kwargs': {'lr': 1e-4},
            'features': (1024, 1024),
        }
        exploration_freq = [[1, init_exploration_freq], [exploitation_switch_at, -1]]
        intrinsic_reward_model = DisagreementIntrinsicReward
        algorithm = DrQv2ExploitAndPlay(
            env=vec_env,
            seed=seed,
            ensemble_model_kwargs=ensemble_model_kwargs,
            intrinsic_reward_model=intrinsic_reward_model,
            train_ensemble_with_target=train_ensemble_with_target,
            update_encoder_with_exploration_policy=update_encoder_with_exploration_policy,
            predict_img_embed=predict_img_embed,
            action_noise=action_noise,
            exploration_freq=exploration_frequency_schedule(
                exploration_freq
            ),
            actor_tau=1.0,
            **algorithm_kwargs
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
        record_video=bool(args.record_video),
        encoder_feature_dim=args.encoder_feature_dim,
        exploitation_switch_at=args.exploitation_switch_at,
        init_exploration_freq=args.init_exploration_freq,
        train_ensemble_with_target=bool(args.train_ensemble_with_target),
        update_encoder_with_exploration_policy=bool(args.update_encoder_with_exploration_policy),
        seed=args.seed,
        task=args.task,
        predict_img_embed=bool(args.predict_img_embed),
        action_cost=args.action_cost,
        sig=args.sig,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTTest')

    # general experiment args
    parser.add_argument('--logs_dir', type=str, default='./logs/')
    parser.add_argument('--project_name', type=str, default='CartPoleImgTest')
    parser.add_argument('--alg', type=str, default='DrQv2')
    parser.add_argument('--total_steps', type=int, default=1_000_000)
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--record_video', type=int, default=0)
    parser.add_argument('--encoder_feature_dim', type=int, default=50)
    parser.add_argument('--train_ensemble_with_target', type=int, default=1)
    parser.add_argument('--update_encoder_with_exploration_policy', type=int, default=0)
    parser.add_argument('--predict_img_embed', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task', type=str, default='swingup_sparse')
    parser.add_argument('--exploitation_switch_at', type=float, default=0.25)
    parser.add_argument('--action_cost', type=float, default=0.0)
    parser.add_argument('--sig', type=float, default=0.2)
    parser.add_argument('--init_exploration_freq', type=int, default=4)
    parser.add_argument('--exp_result_folder', type=str, default=None)

    args = parser.parse_args()
    main(args)
