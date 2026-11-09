# Diabetes Health Indicators Classifier

An educational machine-learning project that predicts the dataset's binary diabetes indicator from health, lifestyle, and demographic survey features. It trains an XGBoost classifier, exposes predictions through a Flask API, and can be run in Docker.


## Problem and intended use

The project asks whether the health indicators in the CDC Behavioral Risk Factor Surveillance System (BRFSS) data can classify a respondent's `Diabetes_binary` label. The intended audience is people exploring an end-to-end classification workflow—from data preparation and evaluation through serving a model—not patients or clinicians.

The target follows the UCI dataset definition: `0` means no diabetes; `1` means prediabetes or diabetes. The model learns associations in survey data; it does not establish cause or diagnose an individual.

## Dataset

The training script downloads `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` from the Kaggle [Diabetes Health Indicators dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset). The dataset is based on CDC BRFSS survey data. The UCI Machine Learning Repository provides the dataset description and variable definitions: [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators).

The selected Kaggle file has 70,692 records, 21 indicator features, and one target column. It is a balanced 50/50 subset. Its class proportions differ from the original survey population, so the reported results and default classification threshold should not be interpreted as population-level prevalence or calibrated individual risk.

The project uses the dataset's health and demographic indicator columns as features and `Diabetes_binary` as the target. Feature names are lowercased during training. The API therefore expects lowercase feature names.

## Method and current results

`train.py` performs stratified train/validation/test splits using a fixed random seed:

- 60% training
- 20% validation
- 20% test

Categorical indicators are converted to strings and one-hot encoded with `DictVectorizer`. Numeric features are passed through as numeric values. The final model is trained with XGBoost using the binary logistic objective and AUC evaluation metric.

The current training run reported:

| Evaluation split | ROC AUC |
| --- | ---: |
| Validation | 0.8308 |
| Test | 0.8297 |

These are results from one seeded split on a balanced dataset, not a guarantee of performance on new populations. The current training script focuses on XGBoost; baseline/model comparison and broader cross-validation can be added as follow-up evaluation.

## Project files

| File | Purpose |
| --- | --- |
| `Project.ipynb` | Data exploration and project notebook |
| `train.py` | Downloads the Kaggle data, prepares features, trains/evaluates the model, and writes `diabetes_model.bin` |
| `predict.py` | Loads the model and serves the `/predict` endpoint |
| `test.py` | Example client request for the local API |
| `Dockerfile` | Builds the API container |
| `Pipfile` / `Pipfile.lock` | Python environment and locked dependencies |

## Run locally

Use Python 3.12, as specified by the `Pipfile`. From the project directory:

```bash
pip install pipenv
pipenv install --deploy
pipenv run python train.py
```

Training downloads the dataset with KaggleHub and saves `diabetes_model.bin` in the project directory. It requires internet access for the initial download.

Start the API in a separate terminal:

```bash
pipenv run python predict.py
```

The service listens on port `9696`.

## API

Send one feature object or a list of feature objects as JSON. Use lowercase names and include the model's 21 features. The response is a class prediction (`0` or `1`) using a fixed probability threshold of 0.5; it is not a probability estimate.

Example request:

```bash
curl -X POST "http://localhost:9696/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "highbp": 1,
    "highchol": 1,
    "cholcheck": 1,
    "bmi": 25,
    "smoker": 0,
    "stroke": 0,
    "heartdiseaseorattack": 0,
    "physactivity": 1,
    "fruits": 1,
    "veggies": 1,
    "hvyalcoholconsump": 0,
    "anyhealthcare": 1,
    "nodocbccost": 0,
    "genhlth": 2,
    "menthlth": 0,
    "physhlth": 0,
    "diffwalk": 0,
    "sex": 0,
    "age": 9,
    "education": 6,
    "income": 8
  }'
```

Example response:

```json
{"predictions": 1}
Have prediabetes or diabetes
```

For a list of records, `predictions` is a list of class labels. The API returns HTTP 400 for invalid JSON or a payload that is not an object or list of objects.

Run the included client example while the API is running:

```bash
pipenv run python test.py
```

## Run with Docker

The image uses the checked-in model file, so train the model first if `diabetes_model.bin` is missing or needs to be regenerated.

```bash
docker build -t diabetes-health-indicators .
docker run --rm -p 9696:9696 diabetes-health-indicators
```

Then send the API request above to `http://localhost:9696/predict`.

## Limitations and next steps

- The balanced Kaggle subset does not reflect the original population class prevalence.
- A single random split does not establish generalization across time, geography, or demographic groups.
- The fixed 0.5 threshold is a demonstration default, not a clinically selected operating point.
- The endpoint returns a binary class, not a calibrated probability or explanation.
- This project does not establish clinical validity, fairness, or suitability for individual health decisions.
- Future work: compare at least three candidate models, use cross-validation for model selection, report additional metrics (such as precision, recall, and a confusion matrix), and assess calibration and subgroup performance.

## References

- UCI Machine Learning Repository, [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators), DOI: [10.24432/C53919](https://doi.org/10.24432/C53919)
- Kaggle, [Diabetes Health Indicators dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- DataTalks.Club, [Machine Learning Zoomcamp project overview](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/projects/README.md)
- DataTalks.Club, [Project tips and checklist](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/projects/project-tips.md)
