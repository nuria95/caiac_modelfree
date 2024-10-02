from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers.time_limit import TimeLimit
from multimexmf.commons.utils.wrappers import HERGoalEnvWrapper
import wandb
from wandb.integration.sb3 import WandbCallback
from gymnasium.wrappers import NormalizeObservation

from multimexmf.commons.intrinsic_reward_algorithms.sac_exploit_and_play \
    import SacExploitAndPlay
from multimexmf.commons.intrinsic_reward_algorithms.utils import\
    exploration_frequency_schedule, DisagreementIntrinsicReward
import numpy as np
import datetime
import os
import argparse
from multimexmf.commons.utils.dict_utils import recursive_objectify
import mbrl.env.mujoco_envs

from typing import Any, Callable, Dict, Optional, Type, Union

import gymnasium as gym

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.vec_video_recorder import \
    VecVideoRecorder


def make_vec_env(
    env_id: Union[str, Callable[..., gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create a wrapped, monitored ``VecEnv``.
    By default it uses a ``DummyVecEnv`` which is usually faster
    than a ``SubprocVecEnv``.

    :param env_id: either the env ID, the env class or a callable returning an env
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_class: Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
        Note: the wrapper specified by this parameter will be applied after the ``Monitor`` wrapper.
        if some cases (e.g. with TimeLimit wrapper) this can lead to undesired behavior.
        See here for more details: https://github.com/DLR-RM/stable-baselines3/issues/894
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :param wrapper_kwargs: Keyword arguments to pass to the ``Wrapper`` class constructor.
    :return: The wrapped environment
    """
    env_kwargs = env_kwargs or {}
    vec_env_kwargs = vec_env_kwargs or {}
    monitor_kwargs = monitor_kwargs or {}
    wrapper_kwargs = wrapper_kwargs or {}
    assert vec_env_kwargs is not None  # for mypy

    def make_env(rank: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            # For type checker:
            assert monitor_kwargs is not None
            assert wrapper_kwargs is not None
            assert env_kwargs is not None

            if isinstance(env_id, str):
                # if the render mode was not specified, we set it to `rgb_array` as default.
                kwargs = {"render_mode": "rgb_array"}
                kwargs.update(env_kwargs)
                try:
                    env = gym.make(env_id, **kwargs)  # type: ignore[arg-type]
                except TypeError:
                    env = gym.make(env_id, **env_kwargs)
            else:
                env = env_id(**env_kwargs)
                # Patch to support gym 0.21/0.26 and gymnasium
                env = _patch_env(env)

            if seed is not None:
                # Note: here we only seed the action space
                # We will seed the env at the next reset
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(
                rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None and monitor_dir is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                for wrapper, kwargs in zip(wrapper_class, wrapper_kwargs):
                    env = wrapper(env, **kwargs)
                # env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    vec_env = vec_env_cls([make_env(i + start_index)
                          for i in range(n_envs)], **vec_env_kwargs)
    # Prepare the seeds for the first reset
    vec_env.seed(seed)
    return vec_env


def experiment(
    conf
):
    time_string = datetime.datetime.now().strftime("%m_%d_%H%M%S")
    conf.logs_dir = os.path.join(
        conf.logs_dir, conf.experiment_name, time_string)
    # exploitation switch has to be between [0, 1]
    assert conf.train.exploitation_switch >= 0 and \
        conf.train.exploitation_switch <= 1
    # from multimexmf.envs.dm2gym import DMCGym
    tb_dir = conf.logs_dir + '/runs'

    # wandb.tensorboard.patch(root_logdir=tb_dir)
    run = wandb.init(
        dir=conf.logs_dir,
        project=conf.project_name,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        config=conf,
    )

    # vec_env = make_vec_env('MountainCarContinuous-v0', n_envs=train.num_envs, seed=seed, wrapper_class=TimeLimit,
    #                        env_kwargs={'render_mode': 'rgb_array'},
    #                        wrapper_kwargs={'max_episode_steps': 1_000})
    # eval_env = make_vec_env('MountainCarContinuous-v0', n_envs=4, seed=seed + 1000,
    #                         env_kwargs={'render_mode': 'rgb_array'},
    #                         wrapper_class=TimeLimit,
    #                         wrapper_kwargs={'max_episode_steps': 1_000}
    #                         )

    # vec_env = make_vec_env(mbrl.env.mujoco_envs.MujocoFetchReachEnv,
    #                        n_envs=conf.train.num_envs, seed=conf.seed,
    #                        wrapper_class=[TimeLimit, HERGoalEnvWrapper],
    #                        env_kwargs={'render_mode': 'rgb_array'},
    #                        wrapper_kwargs=[{'max_episode_steps': conf.env.trial_length}, {}])

    # eval_env = make_vec_env(mbrl.env.mujoco_envs.MujocoFetchReachEnv,
    #                         seed=conf.seed + 1000,
    #                         env_kwargs={'render_mode': 'rgb_array'},
    #                         wrapper_class=[TimeLimit, HERGoalEnvWrapper],
    #                         wrapper_kwargs=[
    #                             {'max_episode_steps': conf.env.trial_length}, {}]
    #                         )

    vec_env = make_vec_env(mbrl.env.mujoco_envs.DisMujocoFppEnv,
                           n_envs=conf.train.num_envs, seed=conf.seed,
                           wrapper_class=[
                               TimeLimit, HERGoalEnvWrapper, NormalizeObservation],
                           env_kwargs={'render_mode': 'rgb_array', 'num_objects': conf.env.num_objects,
                                       'same_col': conf.env.same_col, 'num_cols': conf.env.num_goals, 'fix_goal': conf.env.fix_goal},
                           wrapper_kwargs=[{'max_episode_steps': conf.env.trial_length}, {}])

    eval_env = make_vec_env(mbrl.env.mujoco_envs.DisMujocoFppEnv,
                            seed=conf.seed + 1000,
                            wrapper_class=[
                                TimeLimit, HERGoalEnvWrapper, NormalizeObservation],
                            env_kwargs={'render_mode': 'rgb_array', 'num_objects': conf.env.num_objects,
                                        'same_col': conf.env.same_col, 'num_cols': conf.env.num_goals, 'fix_goal': conf.env.fix_goal},
                            wrapper_kwargs=[{'max_episode_steps': conf.env.trial_length}, {}])

    eval_env = VecVideoRecorder(venv=eval_env,
                                video_folder=conf.logs_dir + '/videos',
                                record_video_trigger=lambda x: conf.save_video,
                                video_length=conf.env.trial_length-1)
    callback = EvalCallback(eval_env,
                            log_path=conf.logs_dir,
                            best_model_save_path=conf.logs_dir,
                            eval_freq=1_000,
                            n_eval_episodes=conf.train.num_exploit_trials,
                            deterministic=True,
                            render=False,
                            verbose=2)

    algorithm_kwargs = {
        'policy': 'MlpPolicy',
        # 'train_freq': 32,
        # 'gradient_steps': 32,
        # 'learning_rate': 1e-3,
        'verbose': 1,
        'tensorboard_log': f"{tb_dir}/{run.id}",
        'gradient_steps': -1,
        'learning_starts': 500 * conf.train.num_envs,
    }

    ensemble_model_kwargs = {
        'learn_std': False,
        'optimizer_kwargs': {'lr': conf.train.ensemble_lr,
                             'weight_decay': conf.train.ensemble_wd},
        'features': tuple(conf.arch.n_layers * [conf.arch.n_units]),

    }

    if conf.arch.ensemble_type == 'MlpEns':
        from multimexmf.models.pretrain_models import EnsembleMLP
        conf.arch.ensemble_type = EnsembleMLP
    else:
        raise NotImplementedError

    # exploitation switch says after how many steps do you switch to maximizing extrinsic reward.
    # if exploitation switch = 0.75 --> first 25 % of the total steps you maximize intrinsic reward and extrinsic thereafter.
    if conf.alg == 'Disagreement':
        exploration_freq = [[1, 1], [conf.train.exploitation_switch, -1]]
        algorithm = SacExploitAndPlay(
            env=vec_env,
            ensemble_model_kwargs=ensemble_model_kwargs,
            intrinsic_reward_model=DisagreementIntrinsicReward,
            exploration_freq=exploration_frequency_schedule(
                exploration_freq
            ),
            **algorithm_kwargs
        )
    else:
        raise NotImplementedError

    algorithm.learn(
        total_timesteps=conf.train.num_steps,
        callback=[WandbCallback(), callback],
    )


def main(args):
    """"""
    from pprint import pprint
    print(args)
    pprint(args.__dict__)
    print('\n ------------------------------------ \n')

    """ Experiment core """
    np.random.seed(args.seed)

    experiment(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTTest')
    parser.add_argument('--confname', type=str, default='sac.yaml')
    args = parser.parse_args()
    import yaml
    with open(f'experiments/confs/{args.confname}', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = recursive_objectify(config, make_immutable=False)
    main(config)
