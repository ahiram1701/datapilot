from pathlib import Path

import pandas as pd
import pytest

SAMPLES = Path(__file__).resolve().parents[2] / "sample_data"


@pytest.fixture
def housing() -> pd.DataFrame:
    return pd.read_csv(SAMPLES / "viviendas.csv")


@pytest.fixture
def churn() -> pd.DataFrame:
    return pd.read_csv(SAMPLES / "clientes_churn.csv")
