"""Data processing utilities for VTCS analysis."""

import numpy as np
import pandas as pd


def add_derived_columns(df):
    """Add derived columns to the tracking DataFrame.

    Args:
        df (pd.DataFrame): Input tracking DataFrame

    Returns:
        pd.DataFrame: DataFrame with added derived columns
    """
    df = df.copy()

    # Calculate velocity and acceleration magnitudes
    df["v_mag"] = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2)
    df["a_mag"] = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2)

    # Calculate velocity and acceleration angles
    df["v_angle"] = np.arctan2(df["vy"], df["vx"]) * 180 / np.pi
    df["a_angle"] = np.arctan2(df["ay"], df["ax"]) * 180 / np.pi

    # Calculate velocity angle differences
    df["diff_v_angle"] = (
        df.groupby("id")["v_angle"]
        .diff()
        .abs()
        .apply(lambda x: min(x, 360 - x) if pd.notnull(x) else x)
    )

    return df


def circular_average(angles):
    """Calculate circular average of angles.

    Args:
        angles (array-like): Array of angles in degrees

    Returns:
        float: Circular average angle in degrees
    """
    radians = np.deg2rad(angles)
    sin_average = np.mean(np.sin(radians))
    cos_average = np.mean(np.cos(radians))
    average_radian = np.arctan2(sin_average, cos_average)
    average_angle = np.rad2deg(average_radian)
    return average_angle % 360


def is_within_forward_direction(target_direction, dx, dy):
    """Check if direction is within forward cone.

    Args:
        target_direction (float): Target direction in degrees
        dx (float or array): X displacement
        dy (float or array): Y displacement

    Returns:
        bool or array: Whether direction is within forward cone
    """
    angle = np.degrees(np.arctan2(dy, dx))
    relative_angle = (angle - target_direction + 360) % 360
    return ((0 <= relative_angle) & (relative_angle <= 45)) | (
        (315 <= relative_angle) & (relative_angle < 360)
    )


def update_selected_and_length(group):
    """Update selected status based on sequence length.

    Args:
        group (pd.DataFrame): Group DataFrame for a single player

    Returns:
        pd.DataFrame: Updated group with length column and filtered selection
    """
    group = group.sort_values("frame").reset_index(drop=True)
    selected = group["selected"].values
    frames = group["frame"].values

    # Find start and end indices of selected sequences
    diff = np.diff(selected.astype(int), prepend=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    if selected[-1] and len(ends) < len(starts):
        ends = np.append(ends, len(selected) - 1)

    # Calculate sequence lengths
    length = np.zeros(len(selected), dtype=int)
    for s, e in zip(starts, ends):
        idx = np.arange(s, e + 1)
        if np.all(np.diff(frames[idx]) == 1):
            seq_length = e - s + 1
            length[idx] = seq_length

    group["length"] = length

    # Filter out sequences that are too short or too long
    group.loc[(group["length"] < 15) | (group["length"] > 75), "selected"] = False

    return group
