from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder
from stable_baselines3.sac import SAC
from gymnasium.wrappers.time_limit import TimeLimit
from multimexmf.commons.max_entropy_algorithms.max_entropy_sac import MaxEntropySAC
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
        ensemble_type: str = 'MlpEns',
        num_envs: int = 8,
        normalize: bool = False,
        record_video: bool = False,
        ensemble_lr: float = 1e-3,
        ensemble_wd: float = 1e-4,
        entropy_switch: float = 0.5,
        no_rotation: bool = True,
        seed: int = 0,
        sparse: bool = True,
        init_exploration_freq: int = 2,
):
    import os
    os.environ['MUJOCO_GL'] = 'osmesa'

    from tactile_envs.envs import InsertionEnv
    tb_dir = logs_dir + 'runs'

    config = dict(
        alg=alg,
        total_steps=total_steps,
        num_envs=num_envs,
        normalize=normalize,
        record_video=record_video,
        ensemble_lr=ensemble_lr,
        ensemble_wd=ensemble_wd,
        entropy_switch=entropy_switch,
        ensemble_type=ensemble_type,
        no_rotation=no_rotation,
        sparse=sparse,
        init_exploration_freq=init_exploration_freq,
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

    if sparse:
        from multimexmf.envs.insertion_sparse import InsertionSparse
        env = lambda: TimeLimit(InsertionSparse(state_type='privileged',
                                                no_rotation=no_rotation,
                                                tactile_shape=(2, 2)), max_episode_steps=300)
    else:
        env = lambda: TimeLimit(InsertionEnv(state_type='privileged',
                                             no_rotation=no_rotation,
                                             tactile_shape=(2, 2)), max_episode_steps=300)
    vec_env = make_vec_env(env, n_envs=num_envs, seed=seed)

    eval_env = make_vec_env(env, n_envs=num_envs, seed=seed + 1000)

    if record_video:
        callback = EvalCallback(VecVideoRecorder(eval_env,
                                                 video_folder=logs_dir + 'eval/',
                                                 record_video_trigger=lambda x: True,
                                                 ),
                                log_path=logs_dir,
                                best_model_save_path=logs_dir,
                                eval_freq=10_000,
                                n_eval_episodes=5,
                                deterministic=True,
                                render=True)
    else:
        callback = EvalCallback(eval_env,
                                log_path=logs_dir,
                                best_model_save_path=logs_dir,
                                eval_freq=10_000,
                                n_eval_episodes=5,
                                deterministic=True,
                                render=False
                                )

    algorithm_kwargs = {
        'policy': 'MultiInputPolicy',
        # 'train_freq': 32,
        # 'gradient_steps': 32,
        # 'learning_rate': 1e-3,
        'verbose': 1,
        'tensorboard_log': f"{tb_dir}/{run.id}",
        'gradient_steps': -1,
        'learning_starts': 300 * num_envs,
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
    # elif alg == 'REDQ':
    #     algorithm_kwargs.pop('policy')
    #     algorithm_kwargs.pop('gradient_steps')
    #     algorithm = REDQ(
    #         env=vec_env,
    #         seed=seed,
    #         **algorithm_kwargs,
    #     )
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
    elif alg == 'Disagreement':
        from multimexmf.commons.intrinsic_reward_algorithms.utils import DisagreementIntrinsicReward, \
            exploration_frequency_schedule
        from multimexmf.commons.intrinsic_reward_algorithms.sac_exploit_and_play import SacExploitAndPlay
        intrinsic_reward_model = DisagreementIntrinsicReward
        exploration_freq = [[1, init_exploration_freq], [entropy_switch, -1]]
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
    # elif alg == 'MaxEntropyREDQ':
    #     algorithm_kwargs.pop('policy')
    #     algorithm_kwargs.pop('gradient_steps')
    #     algorithm = MaxEntropyREDQ(
    #         env=vec_env,
    #         seed=seed,
    #         ensemble_model_kwargs=ensemble_model_kwargs,
    #         dynamics_entropy_schedule=lambda x: 1.0,
    #         **algorithm_kwargs,
    #     )
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
        no_rotation=bool(args.no_rotation),
        sparse=bool(args.sparse),
        init_exploration_freq=args.init_exploration_freq,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTTest')

    # general experiment args
    parser.add_argument('--logs_dir', type=str, default='./logs/')
    parser.add_argument('--project_name', type=str, default='MCTest')
    parser.add_argument('--alg', type=str, default='Disagreement')
    parser.add_argument('--total_steps', type=int, default=250_000)
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--normalize', type=int, default=0)
    parser.add_argument('--record_video', type=int, default=0)
    parser.add_argument('--ensemble_lr', type=float, default=3e-4)
    parser.add_argument('--ensemble_wd', type=float, default=0.0)
    parser.add_argument('--entropy_switch', type=float, default=0.5)
    parser.add_argument('--ensemble_type', type=str, default='MlpEns')
    parser.add_argument('--no_rotation', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sparse', type=int, default=1)
    parser.add_argument('--init_exploration_freq', type=int, default=1)

    parser.add_argument('--exp_result_folder', type=str, default=None)

    args = parser.parse_args()
    main(args)
