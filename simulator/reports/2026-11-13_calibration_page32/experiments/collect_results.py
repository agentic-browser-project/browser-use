"""Collect compact artifacts without GPU caches or source-file overwrites."""
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
REMOTE='shiqihe@spark00.eecs.umich.edu:/home/shiqihe/workspace/calibration_page32_20261113/'
if __name__=='__main__':
    subprocess.run(['rsync','-az','-e','ssh -o ControlMaster=no -o ControlPath=none -o ConnectTimeout=15',
        '--exclude=layouts/','--exclude=experiments/',
        '--exclude=*.log','--exclude=*.lock','--exclude=__pycache__/',REMOTE,str(ROOT)+'/'],check=True)
