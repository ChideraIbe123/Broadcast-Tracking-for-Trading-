"""Game-state win-probability baseline: LightGBM + a logistic reference.

Trained on play-by-play game state only (score margin, time, period, possession).
This is the bar the tracking-augmented model has to beat.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data_pbp import FEATURES, add_features, load_seasons
from .evaluate import evaluate


@dataclass
class WPModels:
    gbm: object
    logit: object
    features: list[str]

    def predict(self, df: pd.DataFrame, which: str = "gbm") -> np.ndarray:
        X = add_features(df)[self.features]
        if which == "gbm":
            return self.gbm.predict_proba(X)[:, 1]
        return self.logit.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> WPModels:
        import joblib

        return joblib.load(path)


def train(train_seasons: list[int], val_seasons: list[int], sample_frac: float = 1.0, seed: int = 0) -> tuple[WPModels, dict]:
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tr = add_features(load_seasons(train_seasons))
    va = add_features(load_seasons(val_seasons))
    if sample_frac < 1.0:
        tr = tr.sample(frac=sample_frac, random_state=seed)
    Xtr, ytr = tr[FEATURES], tr.home_win
    Xva, yva = va[FEATURES], va.home_win

    gbm = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=200,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0, verbose=-1, random_state=seed,
    )
    gbm.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="binary_logloss", callbacks=[lgb.early_stopping(50, verbose=False)])
    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    logit.fit(Xtr, ytr)

    models = WPModels(gbm, logit, FEATURES)
    report = {
        "n_train_events": int(len(tr)),
        "n_val_events": int(len(va)),
        "train_seasons": train_seasons,
        "val_seasons": val_seasons,
        "gbm": evaluate(yva.to_numpy(), models.predict(va, "gbm"), va.secs_left.to_numpy()),
        "logit": evaluate(yva.to_numpy(), models.predict(va, "logit"), va.secs_left.to_numpy()),
        "naive_home": evaluate(yva.to_numpy(), np.full(len(va), float(ytr.mean())), va.secs_left.to_numpy()),
    }
    return models, report
