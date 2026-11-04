"""Feature engineering and leakage-safe target encoding for Airline Passenger Satisfaction."""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import (
    CATEGORICAL_COLS,
    COMFORT_RATING_COLS,
    CONVENIENCE_RATING_COLS,
    DISTANCE_COL,
    ID_COL,
    RATING_COLS,
    SERVICE_RATING_COLS,
    TARGET_COL,
    TARGET_ENCODING_FOLDS,
    TARGET_ENCODING_SMOOTHING,
)

logger = logging.getLogger(__name__)


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Computes engineered features mirroring original notebook exactly.

    Drops TARGET and ID columns if present, cleans numeric columns, computes:
    - Delays: Total_Delay, Delay_Difference, Any_Delay
    - Distance: Log_Flight_Distance
    - Ratings: Rating_Mean, Min, Max, Std, Low/High/Zero counts
    - Aggregate scores: Comfort_Score, Convenience_Score, Service_Score
    - Interactions: Comfort_x_Service, Convenience_x_Comfort, Age_x_Distance
    """
    df = frame.drop(columns=[TARGET_COL, ID_COL], errors="ignore").copy()

    # Convert known numeric columns safely if stored as object strings
    for col in df.columns:
        if df[col].dtype == "object":
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() >= max(1, int(0.90 * df[col].notna().sum())):
                df[col] = converted

    # Delay features
    dep_col = "Departure Delay in Minutes"
    arr_col = "Arrival Delay in Minutes"
    if dep_col in df.columns and arr_col in df.columns:
        dep = pd.to_numeric(df[dep_col], errors="coerce")
        arr = pd.to_numeric(df[arr_col], errors="coerce")
        df["Total_Delay"] = dep.fillna(0) + arr.fillna(0)
        df["Delay_Difference"] = dep.fillna(0) - arr.fillna(0)
        df["Any_Delay"] = ((dep.fillna(0) > 0) | (arr.fillna(0) > 0)).astype("int8")

    # Log flight distance
    if DISTANCE_COL in df.columns:
        distance = pd.to_numeric(df[DISTANCE_COL], errors="coerce")
        df["Log_Flight_Distance"] = np.log1p(distance.clip(lower=0))

    # Rating features
    available_ratings = [c for c in RATING_COLS if c in df.columns]
    if available_ratings:
        ratings = df[available_ratings].apply(pd.to_numeric, errors="coerce")
        df["Rating_Mean"] = ratings.mean(axis=1)
        df["Rating_Min"] = ratings.min(axis=1)
        df["Rating_Max"] = ratings.max(axis=1)
        df["Rating_Std"] = ratings.std(axis=1).fillna(0.0)
        df["Low_Rating_Count"] = (ratings <= 2).sum(axis=1)
        df["High_Rating_Count"] = (ratings >= 4).sum(axis=1)
        df["Zero_Rating_Count"] = (ratings == 0).sum(axis=1)

        def group_mean(cols: List[str]) -> pd.Series:
            present_cols = [c for c in cols if c in ratings.columns]
            if present_cols:
                return ratings[present_cols].mean(axis=1)
            return pd.Series(np.nan, index=df.index)

        df["Comfort_Score"] = group_mean(COMFORT_RATING_COLS)
        df["Convenience_Score"] = group_mean(CONVENIENCE_RATING_COLS)
        df["Service_Score"] = group_mean(SERVICE_RATING_COLS)

        df["Comfort_x_Service"] = df["Comfort_Score"] * df["Service_Score"]
        df["Convenience_x_Comfort"] = df["Convenience_Score"] * df["Comfort_Score"]

    # Age x Distance interaction
    if "Age" in df.columns and DISTANCE_COL in df.columns:
        df["Age_x_Distance"] = (
            pd.to_numeric(df["Age"], errors="coerce")
            * pd.to_numeric(df[DISTANCE_COL], errors="coerce")
        )

    return df


class DistanceTargetEncoder:
    """Leakage-safe smoothed target encoder for route proxy (Flight Distance)."""

    def __init__(
        self,
        smoothing: float = TARGET_ENCODING_SMOOTHING,
        folds: int = TARGET_ENCODING_FOLDS,
        seed: int = 42,
    ):
        self.smoothing = smoothing
        self.folds = folds
        self.seed = seed
        self.global_prior: float = 0.5
        self.mapping: Dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(self, values: pd.Series, target: pd.Series) -> "DistanceTargetEncoder":
        """Fits smoothed target encoding solely on training values and target."""
        val_s = pd.Series(values).reset_index(drop=True)
        tgt_s = pd.Series(target).reset_index(drop=True)

        self.global_prior = float(tgt_s.mean())
        stats = (
            pd.DataFrame({
                "key": val_s.astype(str).values,
                "target": tgt_s.values,
            })
            .groupby("key")["target"]
            .agg(["mean", "count"])
        )

        smooth_scores = (
            stats["mean"] * stats["count"] + self.global_prior * self.smoothing
        ) / (stats["count"] + self.smoothing)
        self.mapping = smooth_scores.to_dict()
        self.is_fitted = True
        return self

    def transform(self, values: pd.Series) -> pd.Series:
        """Transforms distance values using fitted training mapping with prior fallback."""
        if not self.is_fitted:
            raise RuntimeError("DistanceTargetEncoder must be fitted before transforming.")
        val_s = pd.Series(values).astype(str)
        return val_s.map(self.mapping).fillna(self.global_prior)

    def fit_transform_oof(
        self,
        values: pd.Series,
        target: pd.Series,
    ) -> pd.Series:
        """Computes Out-Of-Fold (OOF) target encoding to prevent leakage during training."""
        val_s = pd.Series(values).reset_index(drop=True)
        tgt_s = pd.Series(target).reset_index(drop=True)

        # First fit the global mapping on the full training set for future inference
        self.fit(values, target)

        oof = np.zeros(len(val_s), dtype=float)
        skf = StratifiedKFold(
            n_splits=self.folds,
            shuffle=True,
            random_state=self.seed,
        )

        for fit_idx, hold_idx in skf.split(val_s, tgt_s):
            fold_target = tgt_s.iloc[fit_idx]
            fold_values = val_s.iloc[fit_idx]
            prior = fold_target.mean()

            stats = (
                pd.DataFrame({
                    "key": fold_values.astype(str).values,
                    "target": fold_target.values,
                })
                .groupby("key")["target"]
                .agg(["mean", "count"])
            )
            mapping = (
                stats["mean"] * stats["count"] + prior * self.smoothing
            ) / (stats["count"] + self.smoothing)

            hold_values = val_s.iloc[hold_idx]
            oof[hold_idx] = hold_values.astype(str).map(mapping).fillna(prior).values

        return pd.Series(oof, index=values.index if hasattr(values, "index") else None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "smoothing": self.smoothing,
            "folds": self.folds,
            "seed": self.seed,
            "global_prior": self.global_prior,
            "mapping": self.mapping,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistanceTargetEncoder":
        encoder = cls(
            smoothing=data.get("smoothing", TARGET_ENCODING_SMOOTHING),
            folds=data.get("folds", TARGET_ENCODING_FOLDS),
            seed=data.get("seed", 42),
        )
        encoder.global_prior = data.get("global_prior", 0.5)
        encoder.mapping = data.get("mapping", {})
        encoder.is_fitted = data.get("is_fitted", True)
        return encoder


class TabularDataPipeline:
    """Unified pipeline managing feature engineering, target encoding, and model-ready matrix preparation."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.distance_encoder = DistanceTargetEncoder(seed=seed)
        self.base_columns: List[str] = []
        self.cat_cols: List[str] = []
        self.lgb_encoded_columns: List[str] = []
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TabularDataPipeline":
        """Fits the preprocessor on raw training data and target."""
        X_base = make_features(X)
        self.base_columns = X_base.columns.tolist()

        # Identify categorical columns
        self.cat_cols = X_base.select_dtypes(
            include=["object", "category", "string", "bool"]
        ).columns.tolist()

        # Fit target encoder on distance
        if DISTANCE_COL in X_base.columns:
            self.distance_encoder.fit(X_base[DISTANCE_COL], y)

        # Prepare dummy columns for LightGBM
        X_lgb = X_base.copy()
        for col in self.cat_cols:
            X_lgb[col] = X_lgb[col].fillna("__MISSING__").astype(str)

        if DISTANCE_COL in X_lgb.columns:
            X_lgb["Distance_Target_Encoding"] = self.distance_encoder.transform(
                X_lgb[DISTANCE_COL]
            )

        X_lgb_enc = pd.get_dummies(
            X_lgb, columns=self.cat_cols, dummy_na=False, dtype=np.uint8
        )
        self.lgb_encoded_columns = X_lgb_enc.columns.tolist()
        self.is_fitted = True
        return self

    def transform_catboost(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms input into CatBoost format (native categories preserved as strings)."""
        X_feat = make_features(X)
        # Ensure column consistency
        for col in self.base_columns:
            if col not in X_feat.columns:
                X_feat[col] = np.nan
        X_out = X_feat[self.base_columns].copy()

        for col in self.cat_cols:
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna("__MISSING__").astype(str)

        return X_out

    def transform_lightgbm(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms input into LightGBM format (target encoding + one-hot dummies aligned)."""
        if not self.is_fitted:
            raise RuntimeError("TabularDataPipeline must be fitted before transforming.")

        X_feat = make_features(X)
        for col in self.base_columns:
            if col not in X_feat.columns:
                X_feat[col] = np.nan
        X_out = X_feat[self.base_columns].copy()

        for col in self.cat_cols:
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna("__MISSING__").astype(str)

        if DISTANCE_COL in X_out.columns:
            X_out["Distance_Target_Encoding"] = self.distance_encoder.transform(
                X_out[DISTANCE_COL]
            )
        else:
            X_out["Distance_Target_Encoding"] = self.distance_encoder.global_prior

        # One hot encode
        X_enc = pd.get_dummies(
            X_out, columns=self.cat_cols, dummy_na=False, dtype=np.uint8
        )
        # Align with fitted columns
        return X_enc.reindex(columns=self.lgb_encoded_columns, fill_value=0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "distance_encoder": self.distance_encoder.to_dict(),
            "base_columns": self.base_columns,
            "cat_cols": self.cat_cols,
            "lgb_encoded_columns": self.lgb_encoded_columns,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TabularDataPipeline":
        pipe = cls(seed=data.get("seed", 42))
        pipe.distance_encoder = DistanceTargetEncoder.from_dict(
            data["distance_encoder"]
        )
        pipe.base_columns = data.get("base_columns", [])
        pipe.cat_cols = data.get("cat_cols", [])
        pipe.lgb_encoded_columns = data.get("lgb_encoded_columns", [])
        pipe.is_fitted = data.get("is_fitted", True)
        return pipe

