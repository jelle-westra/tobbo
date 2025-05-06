import argparse

from utils import print_status

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str)
    parser.add_argument('--runs', '-r', type=int, default=30)
    parser.add_argument('--full', '-f', action='store_true', default=False)
    return parser.parse_args()

def main():
    args = parse_args()
    print_status(args.name, n_runs=args.runs, full=args.full)

if (__name__ == '__main__') : 
    main()