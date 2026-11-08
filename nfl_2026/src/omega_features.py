import pandas as pd
import numpy as np

from scipy.stats import entropy
from scipy.ndimage import label


def extract_omega_features(
    attempt_df,
    turn_threshold_deg=30,
    hard_turn_threshold_deg=60,
    entropy_bins=20,
):
    """
    Extract all omega-derived features from a single drill attempt.

    Parameters
    ----------
    attempt_df : pd.DataFrame
        One drill attempt sorted by time.
        Must contain:
            - dir (degrees)

    fps : int
        Tracking frequency (default 10 Hz)

    turn_threshold_deg : float
        Threshold for defining a turn episode.

    hard_turn_threshold_deg : float
        Threshold for defining a hard turn.

    entropy_bins : int
        Number of histogram bins for entropy.

    Returns
    -------
    pd.Series
        One row of omega-based features.
    """

    dt = 0.1

    df = attempt_df.sort_values("time").copy()

    # -----------------------------------------------------
    # Compute omega
    # -----------------------------------------------------

    theta = np.unwrap(np.deg2rad(df["dir"]))

    omega = np.gradient(theta, 0.1)

    omega_deg = np.rad2deg(omega)

    abs_omega = np.abs(omega_deg)

    # -----------------------------------------------------
    # Basic statistics
    # -----------------------------------------------------

    omega_mean = omega_deg.mean()

    omega_abs_mean = abs_omega.mean()

    omega_std = omega_deg.std()

    omega_max = omega_deg.max()

    omega_min = omega_deg.min()

    omega_abs_max = abs_omega.max()

    omega_p90 = np.percentile(abs_omega, 90)

    omega_p95 = np.percentile(abs_omega, 95)

    omega_p99 = np.percentile(abs_omega, 99)


    # -----------------------------------------------------
    # Entropy
    # -----------------------------------------------------

    hist, _ = np.histogram(
        abs_omega,
        bins=entropy_bins
    )

    p = hist / hist.sum()

    p = p[p > 0]

    omega_entropy = entropy(p)

    omega_entropy_norm = (
        omega_entropy /
        np.log(entropy_bins)
    )

    # -----------------------------------------------------
    # Activity metrics
    # -----------------------------------------------------

    turn_activity = (
        abs_omega >
        turn_threshold_deg
    ).mean()

    hard_turn_activity = (
        abs_omega >
        hard_turn_threshold_deg
    ).mean()

    # -----------------------------------------------------
    # Turn episode detection
    # -----------------------------------------------------

    is_turn = (
        abs_omega >
        turn_threshold_deg
    )

    turn_labels, n_turns = label(is_turn)

    turn_durations = []

    for turn_id in range(1, n_turns + 1):

        duration = (
            (turn_labels == turn_id).sum()
            * dt
        )

        turn_durations.append(duration)

    if len(turn_durations):

        mean_turn_duration = np.mean(
            turn_durations
        )

        max_turn_duration = np.max(
            turn_durations
        )

    else:

        mean_turn_duration = 0.0
        max_turn_duration = 0.0

    # -----------------------------------------------------
    # Turning asymmetry
    # -----------------------------------------------------

    right_turns = omega_deg[
        omega_deg > 0
    ]

    left_turns = omega_deg[
        omega_deg < 0
    ]

    right_turn_mean = (
        right_turns.mean()
        if len(right_turns)
        else 0
    )

    left_turn_mean = (
        np.abs(
            left_turns.mean()
        )
        if len(left_turns)
        else 0
    )

    right_turn_max = (
        right_turns.max()
        if len(right_turns)
        else 0
    )

    left_turn_max = (
        np.abs(
            left_turns.min()
        )
        if len(left_turns)
        else 0
    )

    right_turn_fraction = (
        len(right_turns)
        /
        len(omega_deg)
    )

    left_turn_fraction = (
        len(left_turns)
        /
        len(omega_deg)
    )

    turn_bias = abs(
        right_turn_mean
        -
        left_turn_mean
    )

    # -----------------------------------------------------
    # Sustained turning intensity
    # -----------------------------------------------------

    rolling_abs_omega = (
        pd.Series(abs_omega)
        .rolling(
            window=3,
            min_periods=1
        )
        .mean()
    )

    peak_sustained_turn = (
        rolling_abs_omega.max()
    )

    # -----------------------------------------------------
    # Angular acceleration
    # -----------------------------------------------------

    angular_accel = np.gradient(
        omega_deg,
        0.1
    )

    mean_abs_angular_accel = (
        np.abs(
            angular_accel
        ).mean()
    )

    max_abs_angular_accel = (
        np.abs(
            angular_accel
        ).max()
    )

    angular_accel_std = (
        angular_accel.std()
    )

    return pd.Series({

        # Basic omega
        "omega_mean":
            omega_mean,

        "omega_abs_mean":
            omega_abs_mean,

        "omega_std":
            omega_std,

        "omega_max":
            omega_max,

        "omega_min":
            omega_min,

        "omega_abs_max":
            omega_abs_max,

        "omega_p90":
            omega_p90,

        "omega_p95":
            omega_p95,

        "omega_p99":
            omega_p99,

        # Entropy
        "omega_entropy":
            omega_entropy,

        "omega_entropy_norm":
            omega_entropy_norm,

        # Activity
        "turn_activity":
            turn_activity,

        "hard_turn_activity":
            hard_turn_activity,

        # Episodes
        "n_turns":
            n_turns,

        "mean_turn_duration":
            mean_turn_duration,

        "max_turn_duration":
            max_turn_duration,

        # Asymmetry
        "right_turn_mean":
            right_turn_mean,

        "left_turn_mean":
            left_turn_mean,

        "right_turn_max":
            right_turn_max,

        "left_turn_max":
            left_turn_max,

        "right_turn_fraction":
            right_turn_fraction,

        "left_turn_fraction":
            left_turn_fraction,

        "turn_bias":
            turn_bias,

        # Sustained turning
        "peak_sustained_turn":
            peak_sustained_turn,

        # Angular acceleration
        "mean_abs_angular_accel":
            mean_abs_angular_accel,

        "max_abs_angular_accel":
            max_abs_angular_accel,

        "angular_accel_std":
            angular_accel_std,
    })

