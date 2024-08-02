from experiments.utils import generate_run_commands, generate_base_command, dict_permutations
from scripts.examples.drq.finger import experiment as exp
import argparse

PROJECT_NAME = 'FingerDrQv2-23JulImgHard'

_applicable_configs = {
    'total_steps': [3_000_000],
    'num_envs': [8],
    'record_video': [0],
    'encoder_feature_dim': [50],
    'seed': list(range(5)),
    'project_name': [PROJECT_NAME],
    'task': ['turn_hard'],
    'action_cost': [0.0]
}

_applicable_configs_disagreement_drq = {'alg': ['DisagreementDrQ'],
                                        'exploitation_switch_at': [0.5],
                                        'init_exploration_freq': [2, 4],
                                        'train_ensemble_with_target': [1],
                                        'update_encoder_with_exploration_policy': [0],
                                        'predict_img_embed': [0],
                                        } | _applicable_configs
_applicable_configs_drq = {'alg': ['DrQ'],
                           'exploitation_switch_at': [0.75],
                           'init_exploration_freq': [1],
                           'train_ensemble_with_target': [0],
                           'update_encoder_with_exploration_policy': [0]
                           } | _applicable_configs

_applicable_configs_drqv2 = {'alg': ['DrQv2'],
                             'exploitation_switch_at': [0.75],
                             'init_exploration_freq': [1],
                             'train_ensemble_with_target': [0],
                             'update_encoder_with_exploration_policy': [0]
                             } | _applicable_configs

_applicable_configs_disagreement_drqv2 = {'alg': ['DisagreementDrQv2'],
                                          'exploitation_switch_at': [0.5],
                                          'init_exploration_freq': [2, 4],
                                          'train_ensemble_with_target': [1],
                                          'update_encoder_with_exploration_policy': [0],
                                          'predict_img_embed': [0],
                                          } | _applicable_configs


all_flags_combinations = dict_permutations(_applicable_configs_drqv2) + \
                         dict_permutations(_applicable_configs_drq) \
                         + dict_permutations(_applicable_configs_disagreement_drqv2) + \
                         dict_permutations(_applicable_configs_disagreement_drq)

def main(args):
    command_list = []
    logs_dir = '../'
    if args.mode == 'euler':
        logs_dir = '/cluster/scratch/'
        logs_dir += 'sukhijab' + '/' + PROJECT_NAME + '/'

    for flags in all_flags_combinations:
        flags['logs_dir'] = logs_dir
        cmd = generate_base_command(exp, flags=flags)
        command_list.append(cmd)

    # submit jobs
    num_hours = 23 if args.long_run else 3
    generate_run_commands(command_list, num_cpus=args.num_cpus, num_gpus=args.num_gpus,
                          mode=args.mode, duration=f'{num_hours}:59:00', prompt=True, mem=16000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_cpus', type=int, default=10, help='number of cpus to use')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--mode', type=str, default='euler', help='how to launch the experiments')
    parser.add_argument('--long_run', default=True, action="store_true")

    args = parser.parse_args()
    main(args)
