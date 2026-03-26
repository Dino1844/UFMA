# UFMA

UFMA is a temporal video understanding and automatic evaluation project built around a unified `ufma/` codebase. The repository includes training, inference, and evaluation pipelines, with the optimized automatic inference workflow already integrated into `ufma/eval/infer_auto.py`.

## Overview

- The main package path is `ufma/`.
- Training, evaluation, and inference scripts have all been updated to use the `ufma` module name.
- The automatic inference pipeline already includes the optimized checker ranking and refinement logic.
- The repository now supports both `Qwen2-VL` and `Qwen2.5-VL`, including a dedicated `Qwen2.5-VL-3B` script set.
- The repository license is provided in `LICENSE`.

## Repository Layout

```text
UFMA/
|-- ufma/                    # Core package
|   |-- eval/                # Inference and evaluation
|   |-- model/               # Model definitions and builders
|   |-- dataset/             # Dataset wrappers and utilities
|   `-- train/               # Training entrypoints
|-- scripts/                 # Training and evaluation scripts
|-- ufma_optim.py            # Thin wrapper for the optimized auto-inference entry
|-- requirements.txt
`-- LICENSE
```

## Installation

```bash
pip install -r requirements.txt
```

`Qwen2.5-VL` support requires the `transformers` version pinned in `requirements.txt`.

Recommended environment setup:

```bash
export PYTHONPATH="./:$PYTHONPATH"
```

For Windows PowerShell:

```powershell
$env:PYTHONPATH = ".;$env:PYTHONPATH"
```

## Inference and Evaluation

Run automatic inference directly:

```bash
python ufma/eval/infer_auto.py \
  --dataset <dataset_name> \
  --split test \
  --pred_path outputs/<run_name> \
  --model_gnd_path <grounder_model_dir> \
  --model_checker_path <checker_model_dir> \
  --model_pla_path <planner_model_dir>
```

You can also use the provided scripts:

```bash
bash scripts/evaluation/eval_auto_7b.sh <dataset_name> test
bash scripts/evaluation/eval_auto_2b.sh <dataset_name> test
bash scripts/evaluation/eval_auto_25_3b.sh <dataset_name> test
```

The optimized `infer_auto.py` pipeline supports the following control arguments:

- `--checker_topk`
- `--obj_lambda`
- `--obj_beta`
- `--cd_rounds`
- `--cd_step_ratio`
- `--cd_min_step`
- `--uncert_margin_thr`
- `--uncert_disagree_thr`
- `--active_dense_ratio`

These options are used for checker-stage coarse ranking, uncertainty estimation, and coordinate descent refinement.

## Training

Example script entrypoints:

```bash
bash scripts/pretrain/pretrain_grounder_2b.sh
bash scripts/pretrain/pretrain_planner_2b.sh
bash scripts/pretrain/pretrain_checker_2b.sh
bash scripts/pretrain/pretrain_grounder_25_3b.sh
bash scripts/pretrain/pretrain_planner_25_3b.sh
bash scripts/pretrain/pretrain_checker_25_3b.sh
```

Fine-tuning examples:

```bash
bash scripts/finetune/finetune_qvhighlights_2b.sh
bash scripts/finetune/finetune_qvhighlights_7b.sh
bash scripts/finetune/finetune_qvhighlights_25_3b.sh
```

## License

Please refer to `LICENSE` for the full license text and redistribution terms.
We retain the original BSD license even though we have made substantial modifications to the original codebase.

## Acknowledgement

Thanks to VideoMind for providing the high-quality base code.
