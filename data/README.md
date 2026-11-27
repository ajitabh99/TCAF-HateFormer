# Datasets

None of the datasets are redistributed in this repository. Obtain each from its original source under its own licence and terms of use.

| Dataset | Used for | Source | Path expected by the notebook (Kaggle) |
|---|---|---|---|
| Multi3Hate (Bui et al., NAACL 2025) | text + image memes, 5 languages | https://github.com/MinhDucBui/Multi3Hate | `/kaggle/input/datasets/anshr8/multi3hate-2025/` (parquet splits + `multi3hate_images/`) |
| UC Berkeley Measuring Hate Speech (Kennedy et al., 2020; Sachdeva et al., 2022) | English text | Hugging Face `ucberkeley-dlab/measuring-hate-speech` | loaded directly with `datasets.load_dataset` |
| HateMM (Das et al., ICWSM 2023) | video frames + audio | https://github.com/hate-alert/HateMM | `/kaggle/input/datasets/anshr8/hatemm-processed-v2/` (metadata + one frame per video), `.../hatemm-embeddings/hatemm_embeddings.npz` (emotion2vec embeddings), `.../hatemm-annotations/HateMM_annotation.csv` |
| CREMA-D (Cao et al., 2014) | acted emotional speech (weak audio supervision) | https://github.com/CheyneyComputerScience/CREMA-D | `/kaggle/input/datasets/ejlok1/cremad/AudioWAV/` |
| HateXplain (Mathew et al., AAAI 2021) | rationale evaluation only (never trained on) | https://github.com/hate-alert/HateXplain | downloaded at run time from GitHub |

**Preprocessing done outside the notebook.** For HateMM, one representative frame per video (`<video_id>.jpg`) and one utterance-level emotion2vec_plus_base embedding per video were extracted in advance. CREMA-D embeddings are extracted inside the notebook (cell V3-A) and keyed by file name.
