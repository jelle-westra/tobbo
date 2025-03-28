import numpy as np

import os
import subprocess
from IPython.display import display, Markdown
from typing import List

def check_package_status() :
    path = os.path.dirname(os.path.abspath(__file__))
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)

    file_changes = subprocess.run(['git', 'diff', '--name-only', path], capture_output=True, text=True).stdout
    untracked_files = subprocess.run(['git', 'ls-files', '--others', '--exclude-standard', path], capture_output=True, text=True).stdout
    
    if file_changes or untracked_files :
        display(Markdown(
            '<div class="alert alert-block alert-danger" style="font-size: 14px; border-radius: 10px">'
            f'<h4>[NO GO] TO @ {commit.stdout}</h4>'
            '<pre>  ⚠️ Changes found in the ./TO package, first commit changes before starting experiments.</pre>'
            '</div>'
        ))
    else: 
        display(Markdown(
            '<div class="alert alert-block alert-success" style="font-size: 14px; border-radius: 10px">'
            f'<h4>[GO] TO @ {commit.stdout}</h4>'
            '<pre>  ✅ No changes found in the ./TO packge.</pre>'
            '</div>'
        ))

def load_convergence_curve(seed: int, run: int=None) -> np.ndarray :
    path = os.path.join(os.path.abspath(''), f'results/{seed}/evals.dat' if not(run) else f'results/{seed}-{run}/evals.dat')
    with open(path, 'r') as handle : lines = handle.readlines()
    f = np.array([float(line.split()[1]) for line in lines])
    illegal = np.array([float(line.split()[2]) > 0 for line in lines])
    f[illegal] = float('inf')
    return f