"""Execute Task 1 in a fresh kernel, preserving all numbered outputs."""
import argparse
import json
import os
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', choices=['smoke', 'full'], default='full')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    work = root / 'Question 1'
    os.environ['PA1_PRESET'] = args.preset
    os.environ['PA1_BATCH_EXECUTION'] = '1'
    os.environ['PA1_CHECKPOINTS'] = str(work / 'checkpoints' / args.preset)
    os.environ['PA1_THREADS'] = '2'
    # A separate evidence folder per preset prevents stale smoke outputs entering the report.
    destination = work / 'results' / 'design'
    if destination.exists():
        import shutil
        old = work / 'results' / f'previous-{args.preset}'
        if old.exists(): shutil.rmtree(old)
        shutil.move(str(destination), str(old))
    notebook = nbformat.read(work / 'Assignment1.ipynb', as_version=4)
    output = work / f'Assignment1-{args.preset}-executed.ipynb'
    client = NotebookClient(notebook, timeout=None, kernel_name=os.environ.get('PA1_KERNEL', 'python3'),
                            resources={'metadata': {'path': str(work)}}, allow_errors=False)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output)
    print('Executed notebook:', output)
    if args.preset == 'full':
        import subprocess
        subprocess.run([sys.executable, str(root / 'build_report.py')], check=True)
    else:
        print('Smoke checks passed. These results are NOT submission evidence.')
