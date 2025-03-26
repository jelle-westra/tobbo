import os
import shutil
import subprocess

exp = 'original'

run = 1
total = 30
no_failed = 0
while (run < total+1) : 
    subprocess.run([
        'python', 'main.py',
        '--run', f'{run}',
        '--budget', '50',
        '--sigma0', '0.25',
    ])
    failed = False
    if os.path.exists(evals_path := f'./results/{run}/evals.dat') :
        with open(evals_path, 'r') as handle : n = len(handle.readlines())
        failed = (n < 10)
    else: 
        failed = True
    if failed :
        shutil.rmtree(f'./results/{run}/')
        no_failed += 1
        continue
    run += 1
print(f'no. failed runs: {no_failed}')