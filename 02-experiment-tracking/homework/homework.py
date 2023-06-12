# %% [markdown]
# # Experiment Tracking

import os
import pickle

# %%
import subprocess
from pathlib import Path

import click
import mlflow
import optuna
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from optuna.samplers import TPESampler
from preprocess_data import run_data_prep
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

PROJECT_DIR = Path().absolute().parent.parent

# %% [markdown]
# ## Download dataset
#
# We'll use the same NYC taxi
# [dataset](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
# We'll use "**Green** Taxi Trip Records".
#
# Download the data for January, February and March 2022.

# %%
DATA_DIR = PROJECT_DIR / "data"
S3_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/"
FILE_NAMES = [
    "green_tripdata_2022-01.parquet",
    "green_tripdata_2022-02.parquet",
    "green_tripdata_2022-03.parquet",
]


def download_data(file_name: str) -> None:
    file_path = DATA_DIR / file_name
    url = S3_URL + file_name

    if not file_path.is_file():
        print("File does not exist, downloading from S3 bucket.")
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)
        subprocess.run(["wget", "-O", file_path, url])
        print(f"File downloaded successfully and saved at {file_path}")
    else:
        print("File already exists.")


for file_name in FILE_NAMES:
    download_data(file_name)

# %% [markdown]
# ## Q2. Preprocess the data

# %% [markdown]
# Run the script `preprocess_data.py` to preprocess the data and save the
# resulting files in the `data/processed` folder.

# %%
ctx = click.Context(run_data_prep)
ctx.invoke(
    run_data_prep, raw_data_path=str(DATA_DIR), dest_path=str(DATA_DIR / "preprocessed")
)

# %% [markdown]
# So what's the size of the saved DictVectorizer file?

# %%
BYTES_IN_KILOBYTES = 1024


def get_file_size(file_path):
    size_in_bytes = os.path.getsize(file_path)
    size_in_kilobytes = size_in_bytes / BYTES_IN_KILOBYTES
    return size_in_kilobytes


print(
    "Size of DictVectorizer:", get_file_size(DATA_DIR / "preprocessed" / "dv.pkl"), "KB"
)

# %% [markdown]
# ## Q3. Train a model with autolog
#
# Modify the  `train.py` script to enable **autologging** with MLflow, execute
# the script and then launch the MLflow UI to check that the experiment run was
# properly tracked.

# %%
def load_pickle(filename: str | Path):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


# Mlflow setup
mlflow.set_tracking_uri(str(PROJECT_DIR / "mlruns"))
mlflow.set_experiment("hw2_local_experiment")
mlflow.sklearn.autolog()

data_path = DATA_DIR / "preprocessed"
X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))


with mlflow.start_run():

    rf = RandomForestRegressor(max_depth=10, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)

    rmse = mean_squared_error(y_val, y_pred, squared=False)

    print(f"Training complete. RMSE: {rmse:.2f}")

    run_id = mlflow.active_run().info.run_id
    print(f"Run ID: {run_id}")

# %%
# Print the logged parameters and metrics
run = mlflow.get_run(run_id)

print("Parameters:")
[print(f"\t{key}: {value}") for key, value in run.data.params.items()]

print("Metrics:")
[print(f"\t{key}: {value}") for key, value in run.data.metrics.items()]


# %% [markdown]
# ## Q4. Tune model hyperparameters
#
# Now let's try to reduce the validation error by tuning the hyperparameters of
# the `RandomForestRegressor` using `optuna`.
#
# We have prepared the script `hpo.py` for this exercise.
#
# Make sure that the validation RMSE is logged to the tracking server for each
# run of the hyperparameter optimization (you will need to add a few lines of
# code to the `objective` function) and run the script without passing any
# parameters.
#
# Start the MLflow Server with:
#
# ```bash
# mlflow server \
#   --backend-store-uri sqlite:///mlruns.db \
#   --default-artifact-root ./mlruns
# ```

# %%
# Set new experiment using sqlite as backend
MLFLOW_TRACKING_URI = PROJECT_DIR / "mlflow.db"
mlflow.set_tracking_uri("sqlite:///" + MLFLOW_TRACKING_URI.as_posix())

# Turn off autologging
mlflow.sklearn.autolog(disable=True)


mlflow.set_experiment("random-forest-hyperopt")


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def run_optimization(data_path: str, num_trials: int):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 50, 1),
            "max_depth": trial.suggest_int("max_depth", 1, 20, 1),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10, 1),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4, 1),
            "random_state": 42,
            "n_jobs": -1,
        }

        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)

        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metric("rmse", rmse)

        return rmse

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)


data_path = str(DATA_DIR / "preprocessed")
num_trials = 10
run_optimization(data_path, num_trials)

# %%
experiment = mlflow.get_experiment_by_name("random-forest-hyperopt")
runs = mlflow.search_runs(experiment.experiment_id)

best_run = runs.loc[runs["metrics.rmse"].idxmin()]
best_run_id = best_run.run_id

print(f"Best run ID: {best_run_id}")
print(f"Best RMSE metrics: {best_run['metrics.rmse']:.2f}")


# %% [markdown]
# ## Q5. Promote the best model to the model registry
#
# Update the script `register_model.py` so that it selects the model with the
# lowest RMSE on the test set and registers it to the model registry.

# %%
HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = [
    "max_depth",
    "n_estimators",
    "min_samples_split",
    "min_samples_leaf",
    "random_state",
    "n_jobs",
]


MLFLOW_TRACKING_URI = PROJECT_DIR / "mlflow.db"
mlflow.set_tracking_uri("sqlite:///" + MLFLOW_TRACKING_URI.as_posix())
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():
        for param in RF_PARAMS:
            params[param] = int(params[param])

        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)

        # Evaluate model on the validation and test sets
        val_rmse = mean_squared_error(y_val, rf.predict(X_val), squared=False)
        mlflow.log_metric("val_rmse", val_rmse)
        test_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)
        mlflow.log_metric("test_rmse", test_rmse)


def run_model_optimizer(data_path: Path, top_n: int = 5):

    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"],
    )
    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.test_rmse ASC"],
    )[0]

    return best_run


best_run = run_model_optimizer(data_path=DATA_DIR / "preprocessed")
mlflow.register_model(
    f"runs:/{best_run.info.run_id}/model",
    "random-forest-model",
)


# %%
# Print the best run ID and the best test RMSE
print(f"Best run ID: {best_run.info.run_id}")
print(f"Best test RMSE: {best_run.data.metrics['test_rmse']:.2f}")
