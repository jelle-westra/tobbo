import matplotlib.pyplot as plt
import os
from glob import glob

def set_plt_template() :
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.rc('font', family='serif', serif="cmr10", size=18)
    plt.rc('mathtext', fontset='cm', rm='serif')
    plt.rc('axes', unicode_minus=False)

    plt.rcParams['axes.formatter.use_mathtext'] = True


def print_status(experiment: str, pbar_length: int=50, n_runs: int=30, full: bool=False) -> None :
    configs = [c for c in glob(os.path.join(experiment, 'results/*/')) if ('ELA' not in c)]
    configs = sorted(configs, key=lambda c : int(c.split('/')[-2].split('-')[0][:-1]))

    n_configs = len(configs)
    for (config_idx, config) in enumerate(configs) : 
        print(f'[{config_idx+1:02d}/{n_configs}] {config}')

    for (config_idx, config) in enumerate(configs) :
        print('_'*pbar_length)
        print(f'[{config_idx+1:02d}/{n_configs}] {config}')
        config_clean = config.split('/')[-2]
        dimension = int(config_clean.split('-')[0][:-1])
        budget = 100*dimension

        print('config :', config_clean)
        print('budget :', budget)

        runs = sorted(glob(os.path.join(config, '*/')), key=lambda s : int(s.split('/')[-2]))
        if (full) :
            for (i, r) in enumerate(runs) : 
                try:
                    # checking how many times the simulation is called
                    with open(os.path.join(r, 'evals.dat'), 'r') as handle :
                        for line in handle : continue
                        n_evals = int(line.split()[0])
                except FileNotFoundError :
                    n_evals = 0
                try:
                    # checking if the optimizer has terminated
                    with open(os.path.join(r, 'run.log'), 'r') as handle :
                        for line in handle : continue
                        status = 'done' if (line.strip() == '[stop]') else 'active'
                except FileNotFoundError :
                    status = 'starting'

                relative_progress = n_evals/budget
                pbar = round(pbar_length*relative_progress) * '='
                pbar += (pbar_length - len(pbar)) * ' '
                print(f'[{i+1:02d}/{n_runs}]' + f'[{pbar}]' + f'{relative_progress*100:.2f}% ' + f'[{status}]')
        else :
            print(f'runs   : [{len(runs):02d}/{n_runs:02d}]')