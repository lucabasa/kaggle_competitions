import numpy as np
import pandas as pd
import re
from sklearn.pipeline import Pipeline

import tubesml as tml
from tubesml.base import BaseTransformer, fit_wrapper, transform_wrapper


def extract_drill_stat_info(columns):
    drills = set()
    stats = set()
    
    for col in columns:
        # Find all uppercase sequences (potential drill names)
        upper_sequences = re.findall(r'[A-Z]+(?:_[A-Z0-9]+)*', col)
        
        if upper_sequences:
            # The first sequence is likely the drill name
            drill_name = '_'.join(upper_sequences)
            drills.add(drill_name)
            
            # Extract everything after the last drill part as stat
            remaining = col[len(drill_name):]
            if remaining.startswith('_'):
                remaining = remaining[1:]
                
            if remaining and not remaining.isupper():
                stats.add(remaining)
    
    return sorted(drills), sorted(stats)


def aggregate_drills(data, correlation_threshold=0.9, verbose=True):
    df = data.copy()
    if verbose:
        print(f"Inital shape: {df.shape}")

    proc_pipe = Pipeline([("impute", tml.DfImputer(strategy="median")),
                      ('corr_filter', CorrelationFilter(threshold=correlation_threshold)),
                      ])
    res = proc_pipe.fit_transform(df[[c for c in df if c!= "nfl_id"]])

    if verbose:
        print(f"Shape after correlation removed: {res.shape}")

    columns = res.columns.tolist()

    # Extract drills and stats
    drill_list, stat_list = extract_drill_stat_info(columns)
    if verbose:
        print(f"\nNumber of drills: {len(drill_list)}")
        print("Drills:", drill_list)
        print(f"\nNumber of stats: {len(stat_list)}")
        print("Stats:", stat_list)

    data_dict = {}
    data_dict["nfl_id"] = df["nfl_id"]
    for stat in stat_list:
        data_dict[f"{stat}_min"] = df[[c for c in df if stat in c]].min(axis=1)
        data_dict[f"{stat}_mean"] = df[[c for c in df if stat in c]].mean(axis=1)
        data_dict[f"{stat}_max"] = df[[c for c in df if stat in c]].max(axis=1)
        data_dict[f"{stat}_std"] = df[[c for c in df if stat in c]].std(axis=1)

    stats_df = pd.DataFrame(data_dict)

    if verbose:
        print(f"Final shape: {stats_df.shape}")

    return stats_df


def get_pca_components(data, correlation_threshold=0.9, variance_ratio=0.8, verbose=True):

    proc_pipe = Pipeline([("impute", tml.DfImputer(strategy="median")),
                      ('corr_filter', CorrelationFilter(threshold=correlation_threshold)),
                      ("scaler", tml.DfScaler()),
                      ("pca", tml.DfPCA(n_components=variance_ratio, random_state=453))])

    res = proc_pipe.fit_transform(data[[c for c in data if c!= "nfl_id"]])
    pca = proc_pipe.named_steps["pca"].PCA
    corr = proc_pipe.named_steps["corr_filter"]

    if verbose:
        print(f"PCA result shape: {res.shape}")
        print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
        print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_)}")

    filtered_cols = corr.get_feature_names_out()

    loadings = pd.DataFrame(pca.components_.T,
                            index=filtered_cols,
                            columns=res.columns)

    res["nfl_id"] = data["nfl_id"]

    if verbose:
        for pc in loadings.columns[:7]:
            print(pc)
            print(loadings[pc].sort_values(ascending=False).head(15))
            print(loadings[pc].sort_values(ascending=False).tail(15))
            print("\n")

    return res, loadings



class CorrelationFilter(BaseTransformer):
    def __init__(self, threshold=0.9):
        super().__init__()
        self.threshold = threshold

    @fit_wrapper
    def fit(self, X, y=None):
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.features_to_drop = [col for col in upper.columns if any(upper[col] > self.threshold)]

        return self

    @transform_wrapper
    def transform(self, X, y=None):
        return X.drop(columns=self.features_to_drop, errors="ignore")