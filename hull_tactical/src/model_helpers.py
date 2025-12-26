import numpy as np
import pandas as pd

from sklearn.base import RegressorMixin, BaseEstimator
from tubesml.base import BaseTransformer, fit_wrapper, transform_wrapper


class DailyModel(BaseEstimator, RegressorMixin):

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X, y=None):
        preds = np.empty(len(X), dtype=float)
        for _, group in X.groupby("date_id"):
            idx = group.index
            preds[idx] = self.model.predict(group)
        return preds


class ColumnSelector(BaseTransformer):
    """
    A scikit-learn compatible transformer to select columns
    either by explicit list or by regex/pattern.
    """
    def __init__(self, sel_columns=None, patterns=None):
        """
        Parameters
        ----------
        columns : list of str, optional
            Explicit list of column names to select.
        pattern : str, optional
            Pattern to match column names (prefix or regex).
        regex : bool, default=False
            If True, interpret `pattern` as a regex.
            If False, interpret `pattern` as a simple prefix.
        """
        self.sel_columns = sel_columns
        self.patterns = patterns if patterns is not None else []

    @fit_wrapper
    def fit(self, X, y=None):
        # Nothing to fit, just return self
        return self

    @transform_wrapper
    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("ColumnSelector requires a pandas DataFrame as input")

        selected_cols = []

        # Case 1: explicit list
        if self.sel_columns is not None:
            selected_cols.extend(self.sel_columns)

        # Case 2: pattern
        for pattern, match_type in self.patterns:
            if match_type == "startswith":
                selected_cols.extend([c for c in X.columns if c.startswith(pattern)])
            elif match_type == "endswith":
                selected_cols.extend([c for c in X.columns if c.endswith(pattern)])
            elif match_type == "contains":
                selected_cols.extend([c for c in X.columns if pattern in c])
            elif match_type == "exact":
                selected_cols.extend([c for c in X.columns if c == pattern])
            else:
                raise ValueError(f"Unknown match_type: {match_type}")

        # Deduplicate while preserving order
        selected_cols = list(dict.fromkeys(selected_cols))

        return X[selected_cols]
