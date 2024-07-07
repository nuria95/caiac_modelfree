from experiments.utils import generate_run_commands, generate_base_command, dict_permutations
from scripts.examples import mountaincar_play as exp
import argparse

PROJECT_NAME = 'MCTest-06April_longer_run'

_applicable_configs = {
    'alg': ['SAC', 'Random', 'Curiosity', 'Disagreement'],
    'total_steps': [100_000],
    'num_envs': [8],
    'normalize': [0],
    'record_video': [0],
    'ensemble_lr': [3e-4],
    'ensemble_wd': [1e-4],
    'exploration_freq': [0.75, 0.5, 0.25],
    'seed': list(range(5)),
    'project_name': [PROJECT_NAME]
}


all_flags_combinations = dict_permutations(_applicable_configs)


def main(args):
    command_list = []
    logs_dir = './'
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
    parser.add_argument('--long_run', default=False, action="store_true")

    args = parser.parse_args()
    main(args)
