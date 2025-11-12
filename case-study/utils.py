from glob import glob
from datetime import datetime
from typing import List, Dict
import os

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import gridspec
from constructors import mmc_constructors, curved_mmc_constructors, honeycomb_constructors

def set_plt_template() -> None:
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.rc('font', family='serif', serif="cmr10", size=18)
    plt.rc('mathtext', fontset='cm', rm='serif')
    plt.rc('axes', unicode_minus=False)

    plt.rcParams['axes.formatter.use_mathtext'] = True

def load_TO_runs(parameterization: str, dimension: int, optimizer: str, path:str, with_constraints: bool=False) -> None:
    runs_str = 'runs-with-constraints' if with_constraints else 'runs'
    fns = glob(os.path.join(
        path,
        f'{runs_str}/{parameterization}/{dimension}D/optimizer-{optimizer}/*')
    )
        
    fns = sorted(fns, key=lambda fn : int(fn.split('/')[-1]))
    budget = lambda d : d*20
    
    n_runs = len(fns)

    evals_full = np.inf*np.ones((n_runs, budget(dimension)))
    best_configs = np.nan*np.zeros((n_runs, dimension))
    fmin = np.inf*np.ones(n_runs)
    time = np.inf*np.ones(n_runs)

    total_evals = []
    sim_evals = []
    evals = []
    times = []

    for (i, fn) in enumerate(fns):
        fn_evals = os.path.join(fn, 'evals.dat')
        if not(os.path.exists(fn_evals)) : continue

        with open(fn_evals, 'r') as handle : 
            lines = [line for line in handle.readlines()]
            
        it_full = np.array([int(line.split()[0]) - 1 for line in lines if not(line.startswith('#'))])
        it = np.array([int(line.split()[1]) - 1 for line in lines if not(line.startswith('#'))])
        f = np.array([float(line.split()[2]) for line in lines if not(line.startswith('#'))])
        if (f.size == 0) : continue
        
        if not(with_constraints):
            f = f[it < budget(dimension)] # capping of `f` @ 1000
            evals_full[i,it[it < budget(dimension)]] = f
        # else:
            # evals_full[i,it_full] = f
            

        fmin[i] = f.min()
        best_configs[i] = np.array([float(xi) for xi in [line for line in lines if not(line.startswith('#'))][f.argmin()].split()[3:]])

        first = lines[1]
        first_dt = datetime.strptime(first.split(']')[1].strip(), '%Y-%m-%d %H:%M:%S.%f')

        meta_lines = [line for line in lines if line.startswith('#')][:10]
        for last in meta_lines[::-1]:
            if last.startswith('#') : break
        last_dt = datetime.strptime(last.split(']')[1].strip(), '%Y-%m-%d %H:%M:%S.%f')

        # total_evals.append(np.c_[it_full, it])
        total_evals.append(it_full[it < budget(dimension)])
        sim_evals.append(it[it < budget(dimension)])
        evals.append(f)

        time[i] = (last_dt - first_dt).total_seconds() / 60

        times.append([])
        meta_lines = [line for line in lines if line.startswith('#')]
        for line in meta_lines:
            if line[2] == '[':
                times[-1].append(datetime.strptime(line.split(']')[1].strip(), '%Y-%m-%d %H:%M:%S.%f'))

        times[-1] = times[-1][:budget(dimension)//100 + 1]


    evals_min = np.minimum.accumulate(evals_full, axis=1)
    evals_min = evals_min[evals_min[:,0] != np.inf]
    return {
        'min' : fmin,
        'evals_raw' : evals,
        'evals_min' : evals_min,
        'best_configs': best_configs,
        'sim_evals': sim_evals,
        'total_evals': total_evals,
        'time[min]': time,
        'times[min]': times
    }

def plot_convergence_per_simulation(
    dim: int,
    optimizers: List[str],
    parameterizations: List[str],
    runs: dict,
    colors: Dict[str,str] = {'CMAES': 'C0', 'HEBO': 'C1', 'DE': 'C2'},
    linestyles: Dict[str,str] = {'guo': '-', 'mmc': '--', 'hex': ':'},
    labels: Dict[str,str] = {'mmc': 'MMC', 'guo': 'Curved MMC', 'hex': 'Honeycomb'}
) :
    fig, ax = plt.subplots(figsize=(10/2,8/2))
    ax.set_title(f'{dim}D (Simulation Budget = {20*dim})', size=18)

    for optimizer in optimizers:
        ax.plot(1,1,alpha=0, label=optimizer)
        for parameterization in parameterizations:

            evals_min = runs[dim][optimizer][parameterization]['evals_min']
            calls = np.arange(evals_min.shape[1])+1
            mean = np.mean(evals_min, axis=0)
            ax.semilogy(
                calls, mean, 
                label=f'{labels[parameterization]}', 
                c=colors[optimizer], ls=linestyles[parameterization]
            )

            std_err = np.std(evals_min, axis=0, ddof=1) / np.sqrt(len(evals_min))
            ax.fill_between(
                calls, mean - std_err, mean + std_err,
                color=colors[optimizer], alpha=.2
            )

            # print(optimizer, parameterization, np.trapz(evals_min.mean(axis=0)/len(evals_min[0])))

    ax.set_xlabel('Simulation Calls')
    ax.set_ylabel('Compliance')
    # ax.legend(loc='upper right', ncol=3)
    ax.set_xlim(1,dim*20)
    ax.set_yticks([.1,1], [.1,1])
    ax.set_ylim(0.075, 1.2)
    ax.grid(True, which='both', alpha=.2)

    fig.tight_layout()
    return (fig, ax)

def plot_convergence_per_total(
    dim: int,
    optimizers: List[str],
    parameterizations: List[str],
    runs: dict,
    colors: Dict[str,str] = {'CMAES': 'C0', 'HEBO': 'C1', 'DE': 'C2'},
    linestyles: Dict[str,str] = {'guo': '-', 'mmc': '--', 'hex': ':'},
    labels: Dict[str,str] = {'mmc': 'MMC', 'guo': 'Curved MMC', 'hex': 'Honeycomb'}
) :
    fig, ax = plt.subplots(figsize=(10/2,8/2))
    ax.set_title(f'{dim}D (Simulation Budget = {dim*20})', size=18)

    for optimizer in optimizers:
        ax.plot(1,1,alpha=0, label=optimizer)
        for parameterization in parameterizations:
            max_evals = max(runs[dim][optimizer][parameterization]['total_evals'], key=lambda x: x[-1])[-1] + 1
            n_runs = min(15, len(runs[dim][optimizer][parameterization]['evals_raw']))

            fmin = np.inf*np.ones((n_runs, max_evals))

            for idx in range(n_runs):
                evals_min = runs[dim][optimizer][parameterization]['evals_raw'][idx]
                total_evals = runs[dim][optimizer][parameterization]['total_evals'][idx]

                fmin[idx, total_evals] = evals_min

            fmin = np.minimum.accumulate(fmin, axis=1)
            mean = np.mean(fmin, axis=0)
            calls = np.arange(max_evals)+1

            ax.loglog(
                calls, mean, 
                label=f'{labels[parameterization]}', 
                c=colors[optimizer], ls=linestyles[parameterization]
            )

            idx = (mean < np.inf)
            std_err = np.std(fmin[:,idx], axis=0, ddof=1) / np.sqrt(n_runs)

            ax.fill_between(
                calls[idx], mean[idx] - std_err, mean[idx] + std_err,
                color=colors[optimizer], alpha=.2
            )

    # ax.legend(loc='upper right', ncol=1)
    # ax.set_yticks([.1,1], [.1,1])
    # ax.set_ylim(0.075, 1.2)
    ax.grid(True, which='both', alpha=.2)
    ax.set_xlabel('Total Evaluations')
    ax.set_ylabel('Compliance')

    fig.tight_layout()
    return (fig, ax)

def plot_final_distributions(
    dim: int,
    optimizers: List[str],
    parameterizations: List[str],
    runs: dict,
):
    linewidth = 4.79167 # [inch]
    # fig, ax = plt.subplots(1, 2, figsize=(10,8), width_ratios=[3,1])

    fig = plt.figure(figsize=(10/2,8))
    gs = gridspec.GridSpec(9, 2, width_ratios=[3, 1])
    ax0 = plt.subplot(gs[:,0])

    ax1 = [plt.subplot(gs[i,1]) for i in range(9)]

    i = 0
    for parameterization in parameterizations:
        for optimizer in optimizers:
            bplot = ax0.boxplot(runs[dim][optimizer][parameterization]['min'], positions=[8-i], vert=False, patch_artist=True)
            bplot['boxes'][0].set_facecolor('w')
            i += 1

    ax0.boxplot([], patch_artist=True, label='Sub 0.1\nMedian Compliance')['boxes'][0].set_facecolor('#d5e2ef')

    ax0.axhline(2.5, c='k', lw=0.8)
    ax0.axhline(5.5, c='k', lw=0.8)

    ax0.set_xscale('log')
    ax0.set_xticks([.1, .2, .5, 1, 2], [.1, .2, .5, 1, 2])
    ax0.grid(True, axis='x', which='major', alpha=.2)


    ax0.set_yticks(range(len(3*optimizers)), 3*optimizers[::-1], size=12, rotation=90, va='center')
    fig.suptitle(f'{dim}D Final Compliance Distributions \n(15 runs, {20*dim} simulations)', size=18)


    fmin = []
    constructor = mmc_constructors[dim]()
    for (i, optimizer) in enumerate(optimizers):
        fmin.append(runs[dim][optimizer]['mmc']['min'].min())
        x = runs[dim][optimizer]['mmc']['best_configs'][runs[dim][optimizer]['mmc']['min'].argmin()]
        constructor.parameterization.update_topology(constructor.topology, x)
        constructor.topology.plot(ax1[i])
        ax1[i].axis('off')

    constructor = curved_mmc_constructors[dim]()
    for (i, optimizer) in enumerate(optimizers):
        fmin.append(runs[dim][optimizer]['guo']['min'].min())
        x = runs[dim][optimizer]['guo']['best_configs'][runs[dim][optimizer]['guo']['min'].argmin()]
        constructor.parameterization.update_topology(constructor.topology, x)
        constructor.topology.plot(ax1[3+i])
        ax1[3+i].axis('off')

    constructor = honeycomb_constructors[dim]()
    for (i, optimizer) in enumerate(optimizers):
        fmin.append(runs[dim][optimizer]['hex']['min'].min())
        x = runs[dim][optimizer]['hex']['best_configs'][runs[dim][optimizer]['hex']['min'].argmin()]
        constructor.parameterization.update_topology(constructor.topology, x)
        constructor.topology.plot(ax1[6+i])
        ax1[6+i].axis('off')

    ax0.set_xlabel('Final Compliance')

    fig.tight_layout()

    for (i, f) in enumerate(fmin):
        ax1[i].text(50, 50, f'{f:.4f}', size=10, ha='center', va='bottom')

    return (fig, ax0, ax1)