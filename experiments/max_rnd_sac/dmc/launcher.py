from experiments.utils import generate_run_commands, generate_base_command, dict_permutations
from experiments.max_rnd_sac.dmc import experiment as exp
import argparse

PROJECT_NAME = 'MaxRNDSAC20Sept'

_applicable_configs = {
    'num_envs': [1],
    'ensemble_lr': [3e-4],
    'ensemble_wd': [0.0],
    'seed': list(range(5)),
    'project_name': [PROJECT_NAME],
}

_applicable_configs_cartpole = {
                                   'domain_name': ['cartpole-swingup_sparse'],
                                   'action_repeat': [2],
                                   'total_steps': [500_000],
                                   'action_cost': [0.0, 0.1, 0.2],
                               } | _applicable_configs

_applicable_configs_walker = {
                                 'domain_name': ['walker-run'],
                                 'action_repeat': [2],
                                 'total_steps': [1_000_000],
                                 'action_cost': [0.0, 0.1, 0.2],
                             } | _applicable_configs


all_flags_combinations = dict_permutations(_applicable_configs_cartpole) \
                         + dict_permutations(_applicable_configs_walker)


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
    parser.add_argument('--num_cpus', type=int, default=1, help='number of cpus to use')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--mode', type=str, default='euler', help='how to launch the experiments')
    parser.add_argument('--long_run', default=True, action="store_true")

    args = parser.parse_args()
    main(args)
