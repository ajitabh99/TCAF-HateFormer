# TCAF-HateFormer: Cross-Modal Attention Fusion with Multitask Learning for Multilingual Multimodal Hate Speech Detection

Code, experiment configurations, split identifiers, per-run test predictions and integrity audit for the paper:

> A. Mahalkari, P. Yadav, S. Solanki and G. Raghuwanshi, "Cross Modal Attention Fusion with Multitask Learning for Multilingual Multimodal Hate Speech Detection," *Discover Computing* (Springer), 2026.

Every number in the paper's result tables can be checked from the files in this repository **without a GPU** (see *Verify the reported results*). Re-training from scratch needs the four public datasets and a single 16 GB GPU (about 7.5 h for all 38 runs).

> **Naming.** The model is called **TCAF-HateFormer** in the paper. Inside the notebook it still carries its working name *MTL-UEARDNet-v2*, and the file name keeps an early spelling (`ctat-hatefromer`). They are the same model.

## Repository layout

```
notebooks/ctat-hatefromer_V3.2_corrected_pipeline.ipynb   full pipeline: data, splits, model, all 38 training runs, XAI, attacks
audit/split_reconstruction_and_leakage_audit.ipynb       rebuilds the exact splits (CPU, ~5 min) and counts overlaps
configs/experiments.yaml        the 14 configurations x seeds reported in the paper
configs/training.yaml           shared hyper-parameters, split rules and evaluation protocol
splits/                         split identifiers for every dataset (written by the audit notebook)
results/preds/                  test-set predictions of every run: <config>__s<seed>__<m3h|hmm|ucb>.npz
results/runs/                   per-run records: settings, training history, test metrics, bootstrap CIs
results/tables/                 aggregate tables written by the training notebook
results/xai/                    HateXplain plausibility, SHAP/LIME examples, adversarial results
results/audit.json              data-pipeline checks recorded during the training run
scripts/reproduce_tables.py     rebuilds Tables 6-9 from results/preds and checks them against results/tables
```

## Verify the reported results (CPU only, ~3-4 min)

```bash
pip install numpy pandas scikit-learn
python scripts/reproduce_tables.py
```

The script recomputes every per-seed metric and every paired bootstrap comparison (same resampling seed as the notebook) from the raw prediction files and reports the largest difference from the notebook's own tables. Expected output: differences of order 1e-16 (floating-point rounding).

| Paper | Source in this repository |
|---|---|
| Table 6 (all systems, 3 test sets) and Table 9 (ablations) | `results/reproduced/table6_table9_mean_std.csv` |
| Table 7, Figs. 11 and 13 (paired differences, 95% CIs) | `results/reproduced/table7_paired_differences.csv` |
| Table 8 (auxiliary heads H2, H3) | `results/reproduced/table8_aux_heads.csv` |
| Table 10 (adversarial robustness) | `results/xai/table9_adversarial.csv` |
| Table 11 (SHAP / LIME examples) | `results/xai/table10_shap_lime_examples.csv` |
| Table 12 (agreement with HateXplain rationales) | `results/xai/table_plausibility_hatexplain.csv`, `results/xai/plausibility_per_post.csv` |
| Table 13 (per-language Multi3Hate) | `results/tables/v3_per_language.csv` (config `tcaf_masked`) |

*(The XAI file names keep the table numbers of an earlier draft; the mapping above is authoritative.)*

## Data-leakage controls

| Dataset | Split unit | How it is enforced |
|---|---|---|
| Multi3Hate | Meme ID: all five language versions of a meme stay in one split | `StratifiedGroupKFold` grouped by `Meme ID`; the notebook asserts that test and train/val share no Meme ID |
| UC Berkeley MHS | Comment: annotator ratings are averaged per `comment_id` *before* splitting | stratified 70/10/20 split of unique comments |
| HateMM | Video: frame and audio embedding of a clip always share a split | stratified 70/10/20 split of unique videos; embeddings looked up by `video_id` |
| CREMA-D | Actor (speaker) | `GroupShuffleSplit` grouped by actor; the notebook asserts no actor appears in both splits |

The test sets are used once, after training: checkpoints are selected on the combined validation set only, and the decision threshold is fixed at 0.5. HateXplain is used only to score explanations. The audit notebook rebuilds each split with the same code and seeds, confirms that its test labels match the saved predictions one-for-one, and counts identifier, file-hash and normalised-text overlaps between splits; its output is in `splits/integrity_report.json`.

## Re-running the experiments

1. Create a Kaggle notebook from `notebooks/ctat-hatefromer_V3.2_corrected_pipeline.ipynb` with a T4 GPU and Internet on.
2. Attach the datasets listed in `data/README.md` (paths are set in cell 2).
3. Run All. Outputs go to `/kaggle/working/v32`; finished runs are cached, so an interrupted session resumes where it stopped.

Software versions used are listed in `environment.txt`. Seeds are fixed (42, 43, 44), but GPU non-determinism and fp16 arithmetic mean a re-run reproduces the results closely rather than bit-for-bit; the prediction files here are the ones behind the paper.

## Data

The datasets are not redistributed here; see `data/README.md` for sources and licences.

## Citation

See `CITATION.cff`, or cite the paper above.

## Licence

Code: MIT (see `LICENSE`). Prediction files and tables: CC BY 4.0. The datasets keep their own licences.
