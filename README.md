# AI651 Assignment 1 — final submission

Student: Zain Gulbaz (25280005).
Repository link supplied: https://github.com/ZainGulbaz/DLTS-PA-1

## Submit

1. Extract the submission ZIP and upload its folder contents to your GitHub repository. Set the repository to **public**.
2. Read `report/report.pdf` and submit it on LMS with the repository link. The archive includes the LaTeX source, numbered PDF figures, executed full Task 1 notebook and Task 2 execution notebook.
3. If you have not submitted to the leaderboard, paste the complete contents of `Question 2 - Leaderboard/results/predictions.txt` into the predictions field. Declare **P=87809, E=10**. At most five attempts are permitted; do not tune against leaderboard feedback.

No retraining is needed to use this package. The report's unfinished notes have been replaced by interpretations of the uploaded full-run figures and tables. Its AI disclosure identifies the assistance and subsequent report edits; read it before adopting the report.

## Evidence and reproducibility

- Task 1: `Question 1/Assignment1-full-executed.ipynb` has all 23 code cells executed without error; provenance records preset=full and seed/model_seed=0. Deployment choices were stored before final-test access. All numbered CSV/LaTeX/PDF outputs are under `Question 1/results/design/`.
- Task 2: all four configurations ran across seeds 0/1/2. `runs.csv`, `summary.csv`, paired external ablation, block errors and traces preserve the experiments. Final width=64, ten external factors, fixed seed=0; refit=4 epochs. E counts six selected validation-training epochs plus four refit epochs; 112 tuning epochs are separately recorded.
- `final_model.pt` contains the final Task 2 weights/configuration/scalers. `forecast.csv` and `predictions.txt` contain the same 168 values in chronological order. No hidden-target evaluation or leaderboard score is claimed.
- Model implementations and forecasts were not changed during report review. Only report text, disclosure, report preservation and packaging hygiene were edited. Task 1 full checkpoints and Task 2 ablation weights are excluded from the compact archive; training code, full execution outputs and final Task 2 weights remain.

`build_report.py` preserves the reviewed report when its recorded evidence is unchanged. `build_submission.py` validates final evidence before packaging. To reproduce training, install `requirements.txt`, run `run_task1.py --preset full` and `Question 2 - Leaderboard/train_forecast.py --mode full --seeds 0 1 2`. If experiment results change, the report must be updated; the reviewed report is not silently overwritten.

The LMS deadline stated in the conversation is October 5, 2026, 11:59 PM. The October 7 resubmission limit is not presented as a replacement deadline or a penalty-free extension.

`Assignment2-full-executed.ipynb` preserves the actual successful setup and full Task 2 training cells from the uploaded Colab launcher. Their recorded outputs are copied intact, not invented or rerun. The old launcher packaging diagnostic is omitted from the final archive; the uploaded review archive retains it.
