"""Genera datasets sintéticos reproducibles en sample_data/."""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "sample_data"
rng = np.random.default_rng(42)


def housing(n: int = 800) -> pd.DataFrame:
    zona = rng.choice(["centro", "norte", "sur", "periferia"], n, p=[.2, .3, .3, .2])
    area = rng.normal(110, 35, n).clip(35, 300).round()
    recamaras = np.clip((area / 40 + rng.normal(0, .7, n)).round(), 1, 6)
    antiguedad = rng.integers(0, 50, n)
    dist = rng.gamma(2, 4, n).round(1)
    bono = pd.Series(zona).map({"centro": 900_000, "norte": 500_000, "sur": 150_000, "periferia": 0})
    precio = (18_000 * area + 120_000 * recamaras - 15_000 * antiguedad
              - 40_000 * dist + bono + rng.normal(0, 250_000, n)).clip(400_000).round(-3)
    df = pd.DataFrame({"area_m2": area, "recamaras": recamaras, "antiguedad": antiguedad,
                       "distancia_centro_km": dist, "zona": zona, "precio": precio})
    df.loc[rng.choice(n, 20, replace=False), "antiguedad"] = np.nan  # algunos nulos realistas
    return df


def churn(n: int = 1000) -> pd.DataFrame:
    meses = rng.integers(1, 72, n)
    cargo = rng.normal(550, 180, n).clip(150, 1200).round(2)
    contrato = rng.choice(["mensual", "anual", "bianual"], n, p=[.55, .3, .15])
    tickets = rng.poisson(1.5, n)
    logit = (1.2 - .05 * meses + .003 * (cargo - 550) + .45 * tickets
             + pd.Series(contrato).map({"mensual": .9, "anual": -.6, "bianual": -1.4}))
    abandona = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"meses_cliente": meses, "cargo_mensual": cargo, "contrato": contrato,
                         "tickets_soporte": tickets, "abandona": abandona})


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    housing().to_csv(OUT / "viviendas.csv", index=False)
    churn().to_csv(OUT / "clientes_churn.csv", index=False)
    print("Datasets generados en", OUT)
