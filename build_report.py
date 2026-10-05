"""Generate an evidence-backed LaTeX report draft, never inventing missing results."""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent
D=ROOT/'Question 1/results/design'
OUT=ROOT/'report'

def escape(s):
    for a,b in [('\\',r'\textbackslash{}'),('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:
        s=str(s).replace(a,b)
    return s.replace('²','2').replace('<',r'\textless{}').replace('>',r'\textgreater{}')
def table(df):
    return '\n\n'+r'\begin{center}\resizebox{\linewidth}{!}{%'+'\n'+df.to_latex(index=False,escape=True,float_format='%.4f',na_rep='unavailable').replace('²','2')+'}\n'+r'\end{center}'+'\n\n'
def section(title,text): return '\\subsection*{'+title+'}\n'+text+'\n'

def build():
    # Preserve the completed report; verify its data inputs before any regeneration.
    reviewed = OUT / "reviewed_evidence.json"
    if reviewed.exists():
        import hashlib
        hashes = json.loads(reviewed.read_text())["evidence_sha256"]
        for relative, expected in hashes.items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                raise RuntimeError("Evidence changed; update the reviewed report for the new run.")
        print("The reviewed final report is already present; it has not been overwritten.")
        return
    if not D.exists(): raise RuntimeError('Run Task 1 full first.')
    meta=json.loads((D/'run_metadata.json').read_text())
    if meta['preset']!='full': raise RuntimeError('Report generation rejects smoke results. Run full.')
    OUT.mkdir(exist_ok=True)
    cfg=ROOT/'submission_config.json'
    identity=json.loads(cfg.read_text())
    allval=pd.read_csv(D/'all_validation.csv')
    lookup=lambda name,pop: allval[(allval.model==name)&(allval.population==pop)].iloc[0]
    content=section('Response 1A: prediction and measured ordering',
      'Pre-experiment prediction: established stations: period-routed ridge, Raw Attention, '
      'sensor-specific ridge, shared ridge. Station 4: period-routed ridge, Raw Attention, '
      'shared ridge; sensor-specific ridge is unavailable. Rankings are lowest error first. '
      'All results below use the full preset and chronological validation; Station 4 never fitted '
      'the models or preprocessing scale.')
    for pop in ['established','held-out']:
        names=['Shared ridge','Sensor-specific ridge','Period-routed ridge','Raw Attention']
        sub=allval[(allval.population==pop)&allval.model.isin(names)].dropna(subset=['MSE']).sort_values('MSE')
        content+=escape(pop)+': '+escape(' < '.join(sub.model))+'.\\par\n'
        shared=lookup('Shared ridge',pop);routed=lookup('Period-routed ridge',pop)
        content+=f"Shared minus routed validation RMSE: {shared['RMSE (m/s²)']-routed['RMSE (m/s²)']:.4f} m/s2.\\par\n"
    comp=pd.read_csv(D/'1.1.csv')
    for prefix in ['established stations', 'held-out station 4']:
        cols=['model']+[c for c in comp.columns if c.startswith(prefix)]
        compact=comp[cols].copy()
        compact.columns=[c.replace(prefix+', ', '').replace(prefix+' ', '') for c in cols]
        content+=escape(prefix)+' validation comparison.\\par\n'+table(compact)

    content+=('The shared operator compromises across operating paces. Station-specific fitting changes '
      'the routing variable to identity, but a station still experiences many paces; it cannot serve '
      'an unfitted station. Period-routed ridge explicitly supplies the relevant pace assumption, '
      'while Attention learns context-dependent mixing. The measured ordering above revises the hypothesis.\n')
    content+=section('Response 1B: fixed rules and context-built mixing',
      'Ridge uses one fixed matrix for every context; different periods require different phase advances. '
      'Its aggregate error and the individual validation cases in Output 1.2 should be distinguished. '
      'Attention freezes its learned query/key/value projections at inference, but recomputes query--key '
      'scores and the softmax mixing matrix from each input. Output 1.3 establishes context-dependent '
      'mixing, not recovery of the physical period. '
      '\\textbf{Visual review required: identify the specific phase or shape compromise visible in Output 1.2, '
      'and describe the two weight matrices in Output 1.3 before submitting.}')
    content+=table(pd.read_csv(D/'1.2.csv'))
    content+=section('Response 2: moving-average decomposition',
      f"The selected odd width is {meta['selected_kernel']}. Endpoint replication avoids artificial jumps "
      'toward zero; the residual equals input minus trend and reconstructs the input with the trend. '
      'The training-only established-station oracle diagnostic chooses a width against the synthetic '
      'true period. This truth is used for diagnostic selection, not supplied to the neural model. '
      'The 240-sample slow component and 14--36-sample vibration motivate the scale separation. '
      'A centered fixed-width mean can smear an abrupt change and produce boundary distortion. '
      'A robust local median would resist impulses, but imposes piecewise local robustness and lacks '
      'the mean filter\'s linearity. The residual remains an estimate, not the true seasonal signal.')
    content+=table(pd.read_csv(D/'2.2.csv'))
    for pop in ['established','held-out']:
        raw=lookup('Raw Attention',pop);dec=lookup('Attention + decomposition',pop)
        content+=f"{escape(pop)} validation RMSE: raw {raw['RMSE (m/s²)']:.4f}, decomposed {dec['RMSE (m/s²)']:.4f}; "
        content+=f"raw minus decomposed = {raw['RMSE (m/s²)']-dec['RMSE (m/s²)']:.4f} m/s2.\\par\n"
    content+='\\textbf{Visual review required: compare candidate widths and endpoint effects in Output 2.1.}\n'
    recurrence=pd.read_csv(D/'3.2.csv')
    content+=section('Response 3: delay aggregation',
      'The supplied FFT example returns [4, -4, 4, -4]. For values [10,20,30,40], '
      'delays [1,3] and weights [0.75,0.25], aggregation returns [35,15,25,25]. '
      'Indexing is (t-delay) modulo L, so position zero retrieves 40 for delay one. '
      'The diagnostic correlations below are taken from signal values, not learned query/key scores. '
      'Removing slow drift can expose recurrence around the operating period and its multiples. '
      'An isolated transient or a period that changes rapidly inside one window violates a '
      'small recurring-delay assumption: mixing shifted copies can smear an event or align unrelated phases.')
    content+=table(recurrence)
    content+='\\textbf{Visual review required: name the strongest observed peaks in Output 3.2, including any displacement from the marked multiples.}\n'
    content+=section('Response 4: comparison and deployment',
      'Pre-experiment hypothesis: decomposition should help most when slow drift is prominent; '
      'both switches may help, but their gains can overlap. Below are validation results for every '
      'model and population; only the validation-chosen models are evaluated on final test.')
    cols=['model','population','MSE','RMSE (m/s²)','parameters','fit seconds']
    content+=table(allval[cols])
    for pop in ['established','held-out']:
        r=lookup('Raw Attention',pop).MSE;d=lookup('Raw delay mixer',pop).MSE
        a=lookup('Attention + decomposition',pop).MSE;c=lookup('Autoformer-inspired',pop).MSE
        content+=f"{escape(pop)} MSE improvements: decomposition {r-a:.4f}, delay {r-d:.4f}, combined {r-c:.4f}. "
        content+=f"Combined minus sum of individual improvements: {(r-c)-((r-a)+(r-d)):.4f}.\\par\n"
    content+=table(pd.read_csv(D/'4.3.csv'))
    interpretation=(f"The validation-only choices are {meta['choices']['established']} for established stations "
      f"and {meta['choices']['held-out']} for Station 4. The rule minimizes validation RMSE, with exact ties "
      'resolved by parameter count and fitting time. Cost is reported separately; the rule does not optimize '
      'a weighted deployment cost. Positive combined-minus-sum indicates larger-than-additive MSE improvement; '
      'negative indicates overlap. These are empirical differences for one supplied benchmark seed, '
      'not a causal identification of the mechanism. Review Output 4.2 and add a short, evidence-grounded '
      'description of how horizon error changes. Keep the final Response 4 interpretation within 200 words.')
    content+=escape(interpretation)+'\n'
    q2=ROOT/'Question 2 - Leaderboard/results'
    content+='\\section*{Task 2: Autoformer forecasting}\n'
    if (q2/'summary.csv').exists() and (q2/'final_manifest.json').exists():
        summ=pd.read_csv(q2/'summary.csv'); manifest=json.loads((q2/'final_manifest.json').read_text())
        content+=('A compact encoder--decoder Autoformer progressively decomposes hidden states and uses '
          'FFT delay discovery with weighted circular aggregation. External measurements enter the '
          'encoder history and the decoder known horizon. Targets in the decoder horizon are initialized '
          'without future target values. Training-only global scaling preserves level information. '
          'Eight contiguous, nonoverlapping 168-step validation blocks follow the training prefix. '
          'All with/without-external configurations use the same chronological origins and seeds. '
          'Selection minimizes seed-mean validation RMSE; MAE and sMAPE are also reported. '
          'The decoder forecasts the horizon in one pass. Nonnegative projection is evaluated consistently.\n')
        content+=table(summ)
        content+=escape('Final model: '+json.dumps(manifest,sort_keys=True))+'\n'
        content+='The coefficients of the leaderboard cost penalty are undisclosed; validation RMSE and resource counts are reported separately rather than inventing a numerical surrogate score.\n'
    else:
        content+='\\textbf{Task 2 has not yet been run. This report is incomplete for the original assignment scope.}\n'
    content+='\\section*{Evidence figures}\n'
    for f in sorted(D.glob('*.pdf')):
        rel=f.relative_to(ROOT).as_posix()
        content+='\\begin{figure}[p]\\centering\\includegraphics[width=\\linewidth]{\\detokenize{../'+rel+'}}\\caption{Output '+escape(f.stem)+'}\\end{figure}\n'
    content+='\\clearpage\\section*{Attribution and AI disclosure}\n'
    content+=('Task 1 harness and model skeletons were supplied by the course; the three missing '
      'implementations and execution/report tooling were completed with ChatGPT assistance. '
      'Task 2 is an original compact adaptation of the mechanisms in Wu et al. (2021), '
      '\\emph{Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting}, '
      '\\url{https://arxiv.org/abs/2106.13008}. Differences include per-example lag selection, '
      'a compact single-layer default, and explicit known-horizon covariate embeddings.\n')
    content+='\\input{ai_disclosure.tex}\n'
    tex=(r'''\documentclass[10pt]{article}
\usepackage[a4paper,margin=18mm]{geometry}
\usepackage{graphicx,booktabs,longtable,pdflscape,hyperref}
\hypersetup{colorlinks=true,urlcolor=blue}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\begin{document}
\begin{center}\Large AI651 Assignment 1: Temporal Modeling\end{center}
'''+escape(identity['name'])+' --- '+escape(identity['student_id'])+'\\par\nPublic repository: \\url{'+identity['github_url']+'}\\par\n'
      +'\\textbf{Evidence-backed draft: resolve all visual-review notes and confirm Task 2 scope before submission.}\n'
      +'\\section*{Task 1: Forecasting Across Heterogeneous Sensors}\n'+content+'\\end{document}\n')
    (OUT/'report.tex').write_text(tex)
    print('Generated report/report.tex. Replace visual-review notes and set your GitHub URL before compiling.')
if __name__=='__main__': build()
