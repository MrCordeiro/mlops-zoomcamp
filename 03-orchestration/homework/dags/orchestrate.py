import logging
import pathlib
import pickle
from datetime import timedelta

import mlflow
import pandas as pd
import pendulum
from airflow.decorators import dag, task
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

DEFAULT_ARG = {
    "owner": "Airflow",
    "start_date": pendulum.today("UTC").add(days=-2),
    "email": ["your_email@example.com"],
    "email_on_failure": True,
    "email_on_retry": False,
}
PROJECT_DIR = pathlib.Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
THIRD_DAY_OF_MONTH_AT_9AM = "0 9 3 * *"
MLFLOW_TRACKING_URI = PROJECT_DIR / "mlflow.db"
MLFLOW_EXPERIMENT_NAME = "xgboost-03"
MLFLOW_ARTIFACTS_DIR = PROJECT_DIR / "mlruns"


@dag(default_args=DEFAULT_ARG, schedule_interval=THIRD_DAY_OF_MONTH_AT_9AM)
def main_flow() -> None:
    """Orchestrate the ML pipeline"""

    @task(retries=3, retry_delay=timedelta(seconds=2))
    def preprocess_data_and_train_model() -> None:
        # MLflow settings
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_TRACKING_URI}")
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        # Load and transform
        df_train = _add_features(
            pd.read_parquet(DATA_DIR / "green_tripdata_2023-02.parquet")
        )
        df_val = _add_features(
            pd.read_parquet(DATA_DIR / "green_tripdata_2023-03.parquet")
        )

        # Train
        _train_model(df_train, df_val)

    preprocess_data_and_train_model()


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    # duration
    df["duration"] = (
        pd.to_datetime(df["lpep_dropoff_datetime"])
        - pd.to_datetime(df["lpep_pickup_datetime"])
    ).apply(lambda td: td.total_seconds() / 60)
    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]

    # PU_DO
    df[["PULocationID", "DOLocationID"]] = df[["PULocationID", "DOLocationID"]].astype(
        str
    )
    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]
    return df


def _train_model(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
) -> None:
    """Train a model with best hyperparams and write everything out"""

    mlflow.xgboost.autolog()

    best_params = {
        "learning_rate": 0.09585355369315604,
        "max_depth": 30,
        "min_child_weight": 1.060597050922164,
        "objective": "reg:linear",
        "reg_alpha": 0.018060244040060163,
        "reg_lambda": 0.011658731377413597,
        "seed": 42,
    }

    dv = DictVectorizer()
    feature_transformer = ColumnTransformer(
        transformers=[("dv", dv, ["PU_DO", "trip_distance"])], remainder="drop"
    )
    regressor = XGBRegressor(
        early_stopping_rounds=20, num_boost_round=100, **best_params
    )

    pipeline = Pipeline(
        steps=[
            ("feature_transformer", feature_transformer),
            ("regressor", regressor),
        ]
    )

    with mlflow.start_run():
        mlflow.log_params(best_params)

        pipeline.fit(df_train, df_train["duration"])

        y_val = df_val["duration"].values
        y_pred = pipeline.predict(df_val.drop(columns=["duration"]))
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        logging.info(f"RMSE: {rmse}")

        # Save pipeline
        with open(MLFLOW_ARTIFACTS_DIR / "model_pipeline.pkl", "wb") as f:
            pickle.dump(pipeline, f)
        mlflow.log_artifact(str(MLFLOW_ARTIFACTS_DIR / "model_pipeline.pkl"))


main_flow()
