# REF Output Classification Scripts

This repository contains Python scripts converted from Jupyter Notebooks used for fine-tuning and running transformer-based models (e.g. BigBird) to classify UK REF (Research Excellence Framework) outputs by star rating (1\*–4\*). Each script corresponds to a specific Unit of Assessment (UoA).

## Contents

| Script                | Description                                                                                     |
| --------------------- | ----------------------------------------------------------------------------------------------- |
| `Uoa4-chunk-tuned.py` | Fine-tuning and inference for UoA 4 (Psychology), using chunked input to handle long documents. |
| `uoa-4-tuned.py`      | Alternate model for UoA 4, possibly using different preprocessing or training settings.         |
| `uoa-11-tuned.py`     | Model fine-tuned on UoA 11 (Computer Science) REF outputs.                                      |
| `uoa-20-tuned.py`     | Model fine-tuned on UoA 20 (Social Work & Social Policy) REF outputs.                           |

## Prerequisites

* Python 3.9+
* Virtualenv or Conda recommended
* GPU strongly recommended (BigBird model is resource-intensive)

### Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

Typical packages used:

* `transformers`
* `torch`
* `datasets`
* `scikit-learn`
* `pandas`
* `tqdm`

## Usage

Each script follows this general structure:

1. **Data Loading**: Loads tokenized inputs and labels.
2. **Model Definition**: Loads BigBird model and tokenizer (usually `google/bigbird-roberta-base`).
3. **Training Loop**: Fine-tunes the model on REF data using star ratings (1–4).
4. **Evaluation**: Outputs accuracy, F1 score, and confusion matrix.
5. **Inference**: Predicts scores on unseen research outputs.

## Easiest way to run a script is to use Google Colab https://colab.google/ , create a new notebook and upload all datasets needed as given. (Ensure you are connected to a runtime and each filename uploaded matches the name in the code)

## If using Vs code or any other IDE, use the below
To run a script:

```bash
python uoa-11-tuned.py
```

Make sure to modify paths to datasets and checkpoints in the script before running.

## Dataset Format

Scripts assume a preprocessed dataset with the following structure:

* `text`: Full research output (may be chunked if long)
* `label`: Integer label from 0 to 3 (representing 1\* to 4\*)
* Additional metadata like `doi`, `uoa`, or `chunk_id` may be present

## Notes

* The chunked version (`Uoa4-chunk-tuned.py`) handles large documents by splitting them into manageable pieces and averaging logits across chunks during inference.
* Some scripts may require GPU memory optimization (e.g. mixed precision or batch size tuning).
* Class imbalance is addressed using weighting or oversampling where applicable.
