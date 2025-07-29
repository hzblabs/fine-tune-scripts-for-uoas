# REF Output Classification Scripts

This repository contains Python scripts converted from Jupyter Notebooks used for fine-tuning and running transformer-based models (e.g. BigBird) to classify UK REF (Research Excellence Framework) outputs by star rating (1\* to 4\*). Each script corresponds to a specific Unit of Assessment (UoA).

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

## "filename must match code" look at the example below from uoa4-chunk-tuned.py" as it the best model for all depending on the uploaded dataset (i.e uoa4_full_trainingset.jsonl, uoa11_full_trainset.jsonl, uoa ...)

train_file = "uoa4_full_trainset_chunked.jsonl"  <-------- This line of code is in the script, it takes the name of the dataset file, ensure it is correct so it runs.

## These lines of code below are in the prediction test script after model training above (fine tune script)
* zip_path = "uoa4_testset.zip"   <------------------ This takes the zip file of the test set (pdf files you need to test)
* extract_dir = "./uoa4_testset/uoa4_testset"   <---------------- This is where the files get extracted to, ensure the name matches the zip file name
* model_path = "./uoa4_model/checkpoint-1317"   <---------------- This is your model, ensure the path is correct i.e, checkpoint number changes everytime, ensure number is correct
* ground_truth_csv = "uoa4_groundtruth.xlsx"   <---------------- Here is the actual Scores for the pdf files, we upload this to compare the model's score with the groundtruth
* output_csv = "predictions_topk.csv"   <--------------- The model outputs this file to you to see what it predicted.

## If using Vs code or any other IDE, use the below
### Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```
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
