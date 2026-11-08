import pandas as pd

from src.omega_features import extract_omega_features
from src.kinetic_features import extract_speed_accel_features
from src.path_features import extract_path_features


def widen_data(data):
    tmp = data.pivot_table(index="nfl_id",
                           columns="drill_name",
                           values=[c for c in data if c != "nfl_id" and c != "drill_name"])
    tmp.columns = [f'{col[1]}_{col[0]}' if col[1] != '' else col[0] for col in tmp.columns]
    tmp = tmp.reset_index()
    column_nan_percentage = (tmp.isnull().sum() / len(tmp)) * 100
    columns_with_70_percent_data = column_nan_percentage[column_nan_percentage <= 20].index.tolist()
    return tmp[columns_with_70_percent_data]


def preprocess(data):

    df = data[data["entity_type"] == "PLAYER"].copy()

    group_cols = ["nfl_id", "drill_name", "attempt"]

    omega_features = df.groupby(group_cols).apply(extract_omega_features, include_groups=False).reset_index()
    om_agg = {
        'omega_mean': 'mean',
        'omega_abs_mean': 'mean',
        'omega_max': 'max',
        'omega_min': 'min',
        'omega_abs_max': 'max',
        'mean_turn_duration': 'mean',
        'max_turn_duration': 'max',
        'right_turn_mean': 'mean',
        'left_turn_mean': 'mean',
        'right_turn_max': 'max',
        'left_turn_max': 'max',
        'peak_sustained_turn': 'max',
        'mean_abs_angular_accel': 'mean',
        'max_abs_angular_accel': 'max',
        'omega_std': 'mean',
        'angular_accel_std': 'mean',
        'right_turn_fraction': 'mean',
        'left_turn_fraction': 'mean',
        'turn_bias': 'mean',
        'turn_activity': 'max',
        'hard_turn_activity': 'max',
        'n_turns': 'max',
    }
    omega_features = omega_features.groupby(["nfl_id", "drill_name"]).agg(om_agg).reset_index()
    omega_features = widen_data(omega_features)

    speed_acc_featutes = df.groupby(group_cols).apply(extract_speed_accel_features, include_groups=False).reset_index()
    speed_acc_featutes = speed_acc_featutes.groupby(["nfl_id", "drill_name"]).max().reset_index()
    del speed_acc_featutes["attempt"]
    speed_acc_featutes = widen_data(speed_acc_featutes)

    path_features = df.groupby(group_cols).apply(extract_path_features, include_groups=False).reset_index()
    pa_agg = {
        'total_distance': 'max',
        'displacement': 'max',
        'movement_efficiency': 'mean',
        'movement_eccess': 'min',
        'mean_curvature': 'mean',
        'max_curvature': 'max',
        'std_curvature': 'mean',
        'traj_smoothness': 'mean'
    }
    path_features = path_features.groupby(["nfl_id", "drill_name"]).agg(pa_agg).reset_index()
    path_features = widen_data(path_features)

    result = pd.merge(omega_features, speed_acc_featutes, on="nfl_id")
    result = pd.merge(result, path_features, on="nfl_id")

    return result

