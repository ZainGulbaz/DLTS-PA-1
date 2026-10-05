"""Validate final evidence and create a repository-ready archive."""
import json
from pathlib import Path
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parent
config=json.loads((ROOT/'submission_config.json').read_text())
assert 'REPLACE' not in config['student_id'] and 'YOUR_' not in config['github_url'],'Set your identity and repository URL.'
meta=json.loads((ROOT/'Question 1/results/design/run_metadata.json').read_text())
assert meta['preset']=='full','Full-preset evidence is required.'
nb=json.loads((ROOT/'Question 1/Assignment1-full-executed.ipynb').read_text())
for cell in nb['cells']:
    if cell['cell_type']=='code':
        assert cell['execution_count'] is not None,'Every Task 1 code cell must execute.'
        assert not any(o.get('output_type')=='error' for o in cell.get('outputs',[])),'Notebook contains an error.'
for number in ['1.1','1.2','1.3','2.1','2.2','2.3','3.1','3.2','4.1','4.2','4.3']:
    assert (ROOT/f'Question 1/results/design/{number}.csv').exists(),f'Missing Output {number}'
pred=np.array([float(s.strip()) for s in (ROOT/'Question 2 - Leaderboard/results/predictions.txt').read_text().split(',')])
assert pred.shape==(168,) and np.isfinite(pred).all(),'Need exactly 168 finite predictions.'
assert (ROOT/'Question 2 - Leaderboard/results/final_manifest.json').exists()
tex=(ROOT/'report/report.tex').read_text()
assert 'Visual review required' not in tex and 'resolve all visual-review' not in tex,'Replace report review notes with your observations.'
assert 'Unfilled report template' not in tex,'Generate the report from full evidence.'
assert 'Evidence-backed draft' not in tex,'Remove draft label after reviewing the report.'
assert (ROOT/'report/report.pdf').is_file(),'Compile the reviewed report.'
assert (ROOT/'report/ai_disclosure.tex').is_file()
assert 'Complete before submission' not in (ROOT/'report/ai_disclosure.tex').read_text(),'Complete your AI edits and verification disclosure.'
output=ROOT.parent/'DL4STG-PA1-submission.zip'
with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(ROOT)
        if any(v in rel.parts for v in ['__pycache__','.ipynb_checkpoints','checkpoints','smoke_results','verification','previous-full','previous-smoke']):continue
        if p.name == 'RUN_IN_COLAB.ipynb':continue
        if p.name == '.DS_Store' or (p.name.startswith('tmp') and p.suffix == '.pdf'):continue
        if p.suffix in ['.aux','.log','.out','.toc'] or p.name=='Assignment1-smoke-executed.ipynb':continue
        if 'results' in rel.parts and p.name.startswith('w') and p.suffix=='.pt':continue
        z.write(p,Path('DL4STG-PA1-submission')/rel)
print(output)
