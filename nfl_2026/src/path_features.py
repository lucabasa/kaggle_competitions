import pandas as pd
import numpy as np


def extract_path_features(attempt_df):

    df = attempt_df.sort_values("time").copy()

    dt = 0.1

    actual_distance = df["dis"].sum()

    displacement = np.sqrt(
        (df["x"].iloc[-1] - df["x"].iloc[0])**2 +
        (df["y"].iloc[-1] - df["y"].iloc[0])**2
        )

    efficiency = displacement / actual_distance

    eccess_distance = actual_distance - displacement


    # Curvature

    dx = np.gradient(df["x"], dt)
    dy = np.gradient(df["y"], dt)
    ddx = np.gradient(dx, dt)
    ddy = np.gradient(dy, dt)

    kappa = abs(dx * ddy - ddx * dy) / ( dx**2 + dy**2)**(3/2)
    mean_curvature = np.mean(kappa)
    max_curvature = np.max(kappa)
    std_curvature = np.std(kappa)
    traj_smoothness = np.std(np.gradient(kappa, dt))



    return pd.Series({
        "total_distance": actual_distance,
        "displacement": displacement,
        "movement_efficiency": efficiency,
        "movement_eccess": eccess_distance,
        "mean_curvature": mean_curvature,
        "max_curvature": max_curvature,
        "std_curvature": std_curvature,
        "traj_smoothness": traj_smoothness
    }
    )