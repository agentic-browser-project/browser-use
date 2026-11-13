"""Collect compact artifacts without copying GPU caches or changing sources."""
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
REMOTE='shiqihe@spark00.eecs.umich.edu:/home/shiqihe/workspace/calibration_page64_20261112/'

if __name__=='__main__':
    subprocess.run(['rsync','-az','--exclude=captures/','--exclude=layouts/',
        '--exclude=experiments/','--exclude=*.log','--exclude=*.lock',
        '--exclude=__pycache__/',REMOTE,str(ROOT)+'/'],check=True)
