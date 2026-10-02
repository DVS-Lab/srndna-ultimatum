#!/usr/bin/env python3
"""Install the pinned CmdStan toolchain outside the source repository."""
import argparse
from pathlib import Path
import shutil
import cmdstanpy

VERSION = '2.40.0'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install-root', type=Path, required=True)
    parser.add_argument('--cores', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.cores <= 40:
        parser.error('cores must be between 1 and 40')
    if cmdstanpy.__version__ != '1.3.0':
        raise RuntimeError('install code/requirements-gu-stan.txt first')
    for binary in ('make', 'g++'):
        if shutil.which(binary) is None:
            raise RuntimeError(f'{binary} is missing; install a C++ build toolchain first')
    root = args.install_root.resolve()
    repo = Path(__file__).resolve().parents[1]
    if root == repo or repo in root.parents or root in repo.parents:
        parser.error('install toolchain outside repository')
    target = root/f'cmdstan-{VERSION}'
    if (target/'bin/stansummary').is_file() and (target/'bin/stanc').is_file():
        print(f'FOUND: {target}')
    else:
        ok = cmdstanpy.install_cmdstan(version=VERSION, dir=str(root),
            cores=args.cores, progress=False, verbose=True)
        if not ok:
            raise RuntimeError('CmdStan installation failed; see compiler output above')
    cmdstanpy.set_cmdstan_path(str(target))
    if cmdstanpy.cmdstan_version() != (2, 40):
        raise RuntimeError('unexpected installed CmdStan version')
    print(f'PASS: CmdStan {VERSION}; use --cmdstan {target}')


if __name__ == '__main__':
    main()
