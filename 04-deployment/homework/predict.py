import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

PROJECT_DIR = Path(__file__).absolute().parents[2]
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "cohorts" / "2023" / "04-deployment" / "homework"
DATA_SOURCE = "https://d37ci6vzurychx.cloudfront.net/trip-data"
CATEGORICAL_COLS = ["PULocationID", "DOLocationID"]


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(year: int, month: int):

    output_file = DATA_DIR / f"predictions_{year}_{month:02d}.parquet"

    dv, model = load_vectorizer_and_model()
    df = read_data(DATA_SOURCE + f"/yellow_tripdata_{year}-{month:02d}.parquet")
    dicts = df[CATEGORICAL_COLS].to_dict(orient="records")
    X_val = dv.transform(dicts)  # type: ignore
    y_pred = model.predict(X_val)

    logger.info("The predicted mean duration is: %.2f", y_pred.mean())

    # df_result is a dataframe with two columns: ride_id and prediction
    df_result = pd.DataFrame(
        {
            "ride_id": f"{year:04d}/{month:02d}_" + df.index.astype("str"),
            "prediction": y_pred,
        }
    )
    df_result.to_parquet(output_file, engine="pyarrow", compression=None, index=False)


def load_vectorizer_and_model() -> tuple[DictVectorizer, LinearRegression]:
    with open(MODEL_DIR / "model.bin", "rb") as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


def read_data(filename):
    df = pd.read_parquet(filename)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].fillna(-1).astype("int").astype("str")
    return df


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--year", type=int, required=True)
    parser.add_argument("-m", "--month", type=int, required=True)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.year, args.month)
