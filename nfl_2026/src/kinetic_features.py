import pandas as pd
import numpy as np

from scipy.stats import entropy
from scipy.ndimage import label


def extract_speed_accel_features(
    attempt_df,
    fps=10,
    entropy_bins=20,
):
    """
    Extract speed- and acceleration-derived features
    from a single drill attempt.

    Required columns:
        s  (speed)
        a  (acceleration)

    Returns
    -------
    pd.Series
    """

    dt = 1 / fps

    df = attempt_df.sort_values("time").copy()

    speed = df["s"].to_numpy()
    accel = df["a"].to_numpy()

    abs_accel = np.abs(accel)

    # =====================================================
    # SPEED FEATURES
    # =====================================================

    max_speed = np.max(speed)

    mean_speed = np.mean(speed)

    median_speed = np.median(speed)

    speed_std = np.std(speed)

    speed_p90 = np.percentile(speed, 90)
    speed_p95 = np.percentile(speed, 95)
    speed_p99 = np.percentile(speed, 99)

    # =====================================================
    # SPEED ENTROPY
    # =====================================================

    hist_speed, _ = np.histogram(
        speed,
        bins=entropy_bins
    )

    p_speed = (
        hist_speed /
        hist_speed.sum()
    )

    p_speed = p_speed[p_speed > 0]

    speed_entropy = entropy(p_speed)

    speed_entropy_norm = (
        speed_entropy /
        np.log(entropy_bins)
    )

    # =====================================================
    # ACCELERATION FEATURES
    # =====================================================

    peak_accel = np.max(accel)

    peak_braking = np.abs(
        np.min(accel)
    )

    mean_accel = np.mean(accel)

    accel_std = np.std(accel)

    accel_p90 = np.percentile(
        abs_accel,
        90
    )

    accel_p95 = np.percentile(
        abs_accel,
        95
    )

    accel_p99 = np.percentile(
        abs_accel,
        99
    )

    # =====================================================
    # POSITIVE / NEGATIVE ACCELERATION
    # =====================================================

    positive_accel = accel[
        accel > 0
    ]

    negative_accel = accel[
        accel < 0
    ]

    mean_positive_accel = (
        positive_accel.mean()
        if len(positive_accel)
        else 0
    )

    mean_negative_accel = (
        np.abs(
            negative_accel.mean()
        )
        if len(negative_accel)
        else 0
    )

    # =====================================================
    # IMPULSES
    # =====================================================

    accel_impulse = (
        positive_accel.sum()
        * dt
    )

    braking_impulse = (
        np.abs(
            negative_accel
        ).sum()
        * dt
    )

    # =====================================================
    # ACCELERATION ENTROPY
    # =====================================================

    hist_accel, _ = np.histogram(
        accel,
        bins=entropy_bins
    )

    p_accel = (
        hist_accel /
        hist_accel.sum()
    )

    p_accel = p_accel[p_accel > 0]

    accel_entropy = entropy(
        p_accel
    )

    accel_entropy_norm = (
        accel_entropy /
        np.log(entropy_bins)
    )

    # =====================================================
    # JERK
    # =====================================================

    jerk = np.gradient(
        accel,
        dt
    )

    abs_jerk = np.abs(jerk)

    mean_abs_jerk = (
        abs_jerk.mean()
    )

    max_abs_jerk = (
        abs_jerk.max()
    )

    jerk_std = (
        jerk.std()
    )

    # =====================================================
    # SMOOTHNESS
    # =====================================================

    smoothness = (
        1 /
        (
            jerk_std
            + 1e-8
        )
    )

    # =====================================================
    # TIME TO SPEED THRESHOLDS
    # =====================================================

    def first_reach_fraction(
        values,
        fraction
    ):

        target = (
            fraction
            *
            np.max(values)
        )

        idx = np.where(
            values >= target
        )[0]

        if len(idx) == 0:
            return np.nan

        return idx[0] * dt

    t50_speed = (
        first_reach_fraction(
            speed,
            0.50
        )
    )

    t75_speed = (
        first_reach_fraction(
            speed,
            0.75
        )
    )

    t90_speed = (
        first_reach_fraction(
            speed,
            0.90
        )
    )

    # =====================================================
    # PEAK SPEED TIMING
    # =====================================================

    peak_speed_time = (
        np.argmax(speed)
        * dt
    )

    # =====================================================
    # SPEED STABILITY
    # =====================================================

    speed_cv = (
        speed_std /
        (mean_speed + 1e-8)
    )

    accel_cv = (
        accel_std /
        (
            np.mean(
                abs_accel
            )
            + 1e-8
        )
    )

    # =====================================================
    # RETURN
    # =====================================================

    return pd.Series({

        # -----------------
        # SPEED
        # -----------------

        "max_speed":
            max_speed,

        "mean_speed":
            mean_speed,

        "median_speed":
            median_speed,

        "speed_std":
            speed_std,

        "speed_cv":
            speed_cv,

        "speed_p90":
            speed_p90,

        "speed_p95":
            speed_p95,

        "speed_p99":
            speed_p99,

        # -----------------
        # ACCELERATION
        # -----------------

        "peak_accel":
            peak_accel,

        "peak_braking":
            peak_braking,

        "mean_accel":
            mean_accel,

        "accel_std":
            accel_std,

        "accel_cv":
            accel_cv,

        "accel_p90":
            accel_p90,

        "accel_p95":
            accel_p95,

        "accel_p99":
            accel_p99,

        # -----------------
        # POSITIVE/NEGATIVE
        # -----------------

        "mean_positive_accel":
            mean_positive_accel,

        "mean_negative_accel":
            mean_negative_accel,

        "accel_impulse":
            accel_impulse,

        "braking_impulse":
            braking_impulse,

        # -----------------
        # ENTROPY
        # -----------------

        "speed_entropy":
            speed_entropy,

        "speed_entropy_norm":
            speed_entropy_norm,

        "accel_entropy":
            accel_entropy,

        "accel_entropy_norm":
            accel_entropy_norm,

        # -----------------
        # JERK
        # -----------------

        "mean_abs_jerk":
            mean_abs_jerk,

        "max_abs_jerk":
            max_abs_jerk,

        "jerk_std":
            jerk_std,

        "smoothness":
            smoothness,

        # -----------------
        # EXPLOSIVENESS
        # -----------------

        "t50_speed":
            t50_speed,

        "t75_speed":
            t75_speed,

        "t90_speed":
            t90_speed,

        "peak_speed_time":
            peak_speed_time,
    })
