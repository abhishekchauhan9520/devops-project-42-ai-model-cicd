from pathlib import Path
import pandas as pd
from sklearn.datasets import make_classification

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
DATA_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)

FEATURES = ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]
TARGET = "target"


def build_dataset(seed: int = 42) -> pd.DataFrame:
    x, y = make_classification(
        n_samples=1200,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_repeated=0,
        random_state=seed,
        class_sep=1.5,
    )
    return pd.DataFrame(x, columns=FEATURES).assign(target=y)


def validate_dataset(df: pd.DataFrame) -> None:
    expected = set(FEATURES + [TARGET])
    assert set(df.columns) == expected, f"unexpected schema: {df.columns.tolist()}"
    assert len(df) >= 1000, "dataset is unexpectedly small"
    assert df[FEATURES].isna().sum().sum() == 0, "feature data contains nulls"
    assert sorted(df[TARGET].unique().tolist()) == [0, 1], "target must be binary"
