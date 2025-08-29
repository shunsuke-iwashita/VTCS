"""Valuing Timing by Counterfactual Scenarios (VTCS) analysis system.

This module provides a comprehensive system for analyzing ultimate frisbee player
movements and generating counterfactual scenarios to evaluate timing decisions.
It integrates data processing, movement detection, scenario generation, and evaluation
capabilities.
"""

import itertools
import os
import sys

import numpy as np
import pandas as pd


# Data utility functions
def add_derived_columns(df):
    """Add derived columns to the tracking DataFrame.

    Calculates velocity magnitudes, acceleration magnitudes, velocity angles,
    acceleration angles, and velocity angle differences for movement analysis.

    Args:
        df (pd.DataFrame): Input tracking DataFrame containing columns:
            - vx, vy: Velocity components
            - ax, ay: Acceleration components
            - id: Player/object identifier

    Returns:
        pd.DataFrame: DataFrame with added derived columns:
            - v_mag: Velocity magnitude (m/s)
            - a_mag: Acceleration magnitude (m/s²)
            - v_angle: Velocity angle in degrees
            - a_angle: Acceleration angle in degrees
            - diff_v_angle: Absolute velocity angle difference between frames

    Note:
        Velocity angle differences are calculated per player/object ID and
        handle wraparound at 360°/0° boundary.
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

    Computes the circular mean of a set of angles, properly handling
    the wraparound at 360°/0° boundary.

    Args:
        angles (array-like): Array of angles in degrees.

    Returns:
        float: Circular average angle in degrees, normalized to [0, 360).

    Note:
        Uses trigonometric approach by converting to unit vectors,
        averaging, and converting back to angle representation.
    """
    radians = np.deg2rad(angles)
    sin_average = np.mean(np.sin(radians))
    cos_average = np.mean(np.cos(radians))
    average_radian = np.arctan2(sin_average, cos_average)
    average_angle = np.rad2deg(average_radian)
    return average_angle % 360


def is_within_forward_direction(target_direction, dx, dy):
    """Check if direction is within forward cone.

    Determines if a displacement vector (dx, dy) falls within a 45-degree
    forward cone centered on the target direction.

    Args:
        target_direction (float): Target direction in degrees.
        dx (float or np.ndarray): X displacement(s).
        dy (float or np.ndarray): Y displacement(s).

    Returns:
        bool or np.ndarray: Whether direction is within forward cone.
            Returns boolean array if dx/dy are arrays.

    Note:
        Forward cone spans ±22.5 degrees from target direction.
        Uses relative angle calculation to handle wraparound.
    """
    angle = np.degrees(np.arctan2(dy, dx))
    relative_angle = (angle - target_direction + 360) % 360
    return ((0 <= relative_angle) & (relative_angle <= 45)) | (
        (315 <= relative_angle) & (relative_angle < 360)
    )


def update_selected_and_length(group):
    """Update selected status based on sequence length.

    Filters out movement sequences that are too short (< 15 frames) or
    too long (> 75 frames) by analyzing continuous selected sequences.

    Args:
        group (pd.DataFrame): Group DataFrame for a single player containing
            'selected' and 'frame' columns.

    Returns:
        pd.DataFrame: Updated group with added 'length' column and filtered
            'selected' status based on sequence length criteria.

    Note:
        - Identifies continuous sequences of selected frames
        - Calculates length for each sequence
        - Sets selected=False for sequences outside [15, 75] frame range
        - Requires consecutive frame numbers for valid sequences
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


# Movement detection classes
class MovementDetector:
    """Detects and extracts movement candidates from tracking data.

    This class analyzes ultimate frisbee tracking data to identify significant
    player movements that could be used for counterfactual scenario analysis.

    Attributes:
        play (pd.DataFrame): Copy of the input tracking data with additional
            derived columns for movement detection.
    """

    def __init__(self, play_df):
        """Initialize with play DataFrame.

        Args:
            play_df (pd.DataFrame): Tracking data with derived columns including
                velocity magnitudes, angles, and acceleration data.
        """
        self.play = play_df.copy()

    def detect_movement_initiation(self, v_threshold=3.0, a_threshold=4.0):
        """Detect movement initiation and perform temporal expansion.

        Identifies when offensive players begin significant movements based on
        acceleration and velocity criteria, then expands the detection both
        forward and backward in time to capture complete movement sequences.

        Args:
            v_threshold (float): Minimum velocity threshold for movement detection
                in m/s. Defaults to 3.0.
            a_threshold (float): Minimum acceleration threshold for movement
                initiation in m/s². Defaults to 4.0.

        Note:
            This method modifies self.play by adding a 'selected' column and
            removes temporary calculation columns after processing.
        """
        self._detect_initial_movements(v_threshold, a_threshold)
        self._expand_forward(v_threshold)
        self._filter_by_length()
        self._expand_backward()
        self._cleanup_columns()

    def _detect_initial_movements(self, v_threshold, a_threshold):
        """Detect initial movement based on acceleration and velocity criteria.

        Args:
            v_threshold (float): Minimum velocity threshold for movement detection.
            a_threshold (float): Minimum acceleration threshold for movement initiation.

        Note:
            Sets the 'selected' column to True for frames that meet all criteria:
            - Frame number > 1
            - Player is offensive
            - Player is not holding the disc
            - Player was not holding the disc 30 frames ago
            - Acceleration magnitude exceeds threshold
            - Angle between velocity and acceleration vectors < 90 degrees
        """
        self.play["selected"] = False
        self.play["prev_holder"] = self.play.groupby("id")["holder"].shift(30)
        self.play["prev_holder"] = self.play["prev_holder"].fillna(False)

        # Calculate angle difference between velocity and acceleration
        angle_diff = np.abs(
            (self.play["v_angle"] - self.play["a_angle"] + 180) % 360 - 180
        )

        # Define selection criteria
        selection_criteria = (
            (self.play["frame"] > 1)
            & (self.play["class"] == "offense")
            & (~self.play["holder"])
            & (~self.play["prev_holder"])
            & (self.play["a_mag"] > a_threshold)
            & (angle_diff < 90)
        )

        self.play.loc[selection_criteria, "selected"] = True

    def _expand_forward(self, v_threshold):
        """Expand selection forward in time based on velocity consistency.

        For each player, extends the movement selection to subsequent frames
        if the player maintains consistent velocity direction and magnitude.

        Args:
            v_threshold (float): Minimum velocity threshold to continue selection.

        Note:
            Uses circular averaging of velocity angles to determine movement
            consistency. Continues selection if:
            - Velocity angle change ≤ 20 degrees
            - Velocity magnitude > threshold
            - Player is not holding disc
            - Angle difference from average ≤ 90 degrees
        """
        for player_id, player_df in self.play.groupby("id"):
            player_df = player_df.sort_values("frame").reset_index()
            frames = player_df["frame"].values
            v_angles = player_df["v_angle"].values
            selected = player_df["selected"].values
            diff_v_angle = player_df["diff_v_angle"].values
            v_mag = player_df["v_mag"].values
            holder = player_df["holder"].values

            v_angle_history = []
            for i in range(len(frames)):
                frame = frames[i]
                if not selected[i]:
                    v_angle_history = []
                    continue

                v_angle_history.append(v_angles[i])

                # Find next frame
                next_idx = np.where(frames == frame + 1)[0]
                if len(next_idx) == 0:
                    continue
                ni = next_idx[0]

                # Calculate angle consistency
                angle_diff = abs(v_angles[ni] - circular_average(v_angle_history))
                angle_diff = min(angle_diff, 360 - angle_diff)

                # Check continuation conditions
                should_continue = (
                    (diff_v_angle[ni] <= 20)
                    & (v_mag[ni] > v_threshold)
                    & (not holder[ni])
                    & (angle_diff <= 90)
                )

                selected[ni] = should_continue

            self.play.loc[player_df["index"], "selected"] = selected

    def _filter_by_length(self):
        """Filter selections based on sequence length.

        Removes movement sequences that are too short (< 15 frames) or
        too long (> 75 frames) to be considered valid movements.

        Note:
            Applies the update_selected_and_length function from data_utils
            to each player group. This function calculates sequence lengths
            and filters out sequences outside the valid range.
        """
        self.play = self.play.groupby("id", group_keys=False).apply(
            update_selected_and_length
        )

    def _expand_backward(self):
        """Expand selection backward in time based on velocity magnitude.

        Extends movement sequences backward if previous frames show similar
        velocity patterns, capturing the full extent of movement initiation.

        Note:
            Expands backward if:
            - Previous frame is not already selected
            - Previous frame velocity magnitude > 0.05 m/s
            - Velocity magnitude difference < 0.05 m/s
        """
        self.play.set_index(["id", "frame"], inplace=True)

        max_frame = self.play.index.get_level_values("frame").max()
        min_frame = self.play.index.get_level_values("frame").min()

        for frame in range(max_frame, min_frame, -1):
            selected_indices = self.play.index[
                self.play["selected"]
                & (self.play.index.get_level_values("frame") == frame)
            ]

            for idx in selected_indices:
                player_id = idx[0]
                prev_idx = (player_id, frame - 1)

                if prev_idx in self.play.index:
                    if (
                        not self.play.at[prev_idx, "selected"]
                        and self.play.at[prev_idx, "v_mag"] > 0.05
                        and self.play.at[prev_idx, "v_mag"] - self.play.at[idx, "v_mag"]
                        < 0.05
                    ):
                        self.play.at[prev_idx, "selected"] = True

        self.play.reset_index(inplace=True)

    def _cleanup_columns(self):
        """Remove temporary calculation columns.

        Drops columns that were added during movement detection but are not
        needed for subsequent analysis.

        Note:
            Removes: ax, ay, v_mag, a_mag, a_angle, diff_v_angle, prev_holder, length
        """
        columns_to_drop = [
            "ax",
            "ay",
            "v_mag",
            "a_mag",
            "a_angle",
            "diff_v_angle",
            "prev_holder",
            "length",
        ]
        self.play.drop(columns=columns_to_drop, inplace=True)

    def deselect_crowded_movements(self, distance_threshold=5.0, player_threshold=2):
        """Deselect movements in crowded areas or with similar directions.

        Filters out movement candidates that occur in crowded areas or when
        multiple players are moving in the same direction, as these are likely
        less significant individual movements.

        Args:
            distance_threshold (float): Distance threshold for determining crowded
                conditions in meters. Defaults to 5.0.
            player_threshold (int): Minimum number of nearby players for crowded
                condition. Defaults to 2.

        Note:
            Deselection criteria include:
            - Distance threshold of 5.0 meters for determining crowded conditions
            - Player threshold of 2 or more nearby players
            - Forward direction cone of 45 degrees for detecting similar movements
        """
        for player_id in self.play["id"].unique():
            selected_frames = self.play.loc[
                (self.play["id"] == player_id) & (self.play["selected"]), "frame"
            ].values

            # Group continuous frames
            continuous_sequences = [
                list(g)
                for _, g in itertools.groupby(
                    selected_frames, key=lambda n, c=itertools.count(): n - next(c)
                )
            ]

            for sequence in continuous_sequences:
                if not sequence:
                    continue

                last_frame = sequence[-1]
                should_deselect = self._check_crowded_conditions(
                    player_id, last_frame, distance_threshold, player_threshold
                )

                if should_deselect:
                    self.play.loc[
                        (self.play["id"] == player_id)
                        & (self.play["frame"].isin(sequence)),
                        "selected",
                    ] = False

        # Round velocity angles for consistency
        self.play["v_angle"] = self.play["v_angle"].round(2)

    def _check_crowded_conditions(
        self, player_id, frame, distance_threshold, player_threshold
    ):
        """Check if movement should be deselected due to crowded conditions.

        Args:
            player_id (int): ID of the player being evaluated.
            frame (int): Frame number to check.
            distance_threshold (float): Distance threshold for nearby players.
            player_threshold (int): Minimum count for crowded condition.

        Returns:
            bool: True if movement should be deselected due to crowded conditions.

        Note:
            Returns True if either:
            - Number of nearby players ≥ player_threshold
            - Number of players moving in similar direction ≥ 2
        """
        frame_data = self.play[self.play["frame"] == frame]

        # Get target player data
        target_row = frame_data[frame_data["id"] == player_id]
        if target_row.empty:
            return False

        target_x, target_y, target_dir = target_row[["x", "y", "v_angle"]].values[0]

        # Get other offensive players
        others = frame_data[
            (frame_data["id"] != player_id) & (frame_data["class"] == "offense")
        ]

        if others.empty:
            return False

        # Calculate distances and directions
        dx = others["x"].values - target_x
        dy = others["y"].values - target_y
        distances = np.hypot(dx, dy)

        # Count nearby players
        nearby_count = np.sum(distances <= distance_threshold)

        # Count players moving in similar direction
        similar_direction_count = np.sum(
            is_within_forward_direction(target_dir, dx, dy)
        )

        return nearby_count >= player_threshold or similar_direction_count >= 2

    def extract_candidates(self, window=30):
        """Extract movement candidates with temporal windows.

        Extracts continuous sequences of selected movements for each player,
        adding temporal context windows before and after each movement sequence.

        Args:
            window (int): Number of frames to include before and after each
                movement sequence for context. Defaults to 30.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of candidate DataFrames with keys
                in format "{player_id}-{sequence_number}".

        Note:
            The extraction process:
            1. Groups selected movements by player ID
            2. Identifies continuous frame sequences for each player
            3. Adds temporal context windows around each sequence
            4. Creates candidate DataFrames with movement metadata
            5. Marks closest defenders during movement periods
        """
        candidates = {}
        selected_data = self.play[self.play["selected"]]

        for player_id, group in selected_data.groupby("id"):
            frames = group["frame"].sort_values().values
            if len(frames) == 0:
                continue

            # Find continuous sequences
            split_indices = np.where(np.diff(frames) != 1)[0] + 1
            frame_groups = np.split(frames, split_indices)

            for i, frame_sequence in enumerate(frame_groups):
                candidate_df = self._create_candidate_dataframe(
                    frame_sequence, player_id, window
                )
                candidates[f"{player_id}-{i+1}"] = candidate_df

        return candidates

    def _create_candidate_dataframe(self, frame_sequence, player_id, window):
        """Create a candidate DataFrame for a movement sequence.

        Args:
            frame_sequence (np.ndarray): Array of frame numbers for the movement.
            player_id (int): ID of the player performing the movement.
            window (int): Number of context frames to include before and after.

        Returns:
            pd.DataFrame: Candidate DataFrame with movement data and metadata.

        Note:
            Creates candidate DataFrames with attributes:
            - movement_player_id: ID of the moving player
            - movement_start_frame: First frame of movement
            - movement_end_frame: Last frame of movement

            Also sets selected_def=True for the closest defender during movement.
        """
        start_frame = max(frame_sequence.min() - window, self.play["frame"].min())
        end_frame = min(frame_sequence.max() + window, self.play["frame"].max())

        # Extract relevant frames
        candidate_df = self.play[
            (self.play["frame"] >= start_frame) & (self.play["frame"] <= end_frame)
        ].copy()

        # Reset selection flags
        candidate_df["selected"] = False
        candidate_df.loc[
            (candidate_df["id"] == player_id)
            & (candidate_df["frame"].isin(frame_sequence)),
            "selected",
        ] = True

        # Add metadata
        candidate_df.attrs["movement_player_id"] = player_id
        candidate_df.attrs["movement_start_frame"] = frame_sequence.min()
        candidate_df.attrs["movement_end_frame"] = frame_sequence.max()

        # Mark closest defenders
        candidate_df = self._mark_closest_defenders(candidate_df, frame_sequence)

        return candidate_df

    def _mark_closest_defenders(self, candidate_df, frame_sequence):
        """Mark closest defenders during movement period.

        Args:
            candidate_df (pd.DataFrame): Candidate DataFrame to modify.
            frame_sequence (np.ndarray): Array of frames during movement.

        Returns:
            pd.DataFrame: Modified DataFrame with selected_def column updated.

        Note:
            For each frame in the movement sequence, identifies the closest
            defender to the moving player and marks them with selected_def=True.
        """
        candidate_df["selected_def"] = False

        for frame in frame_sequence:
            frame_data = candidate_df[candidate_df["frame"] == frame]
            selected_player = frame_data[frame_data["selected"]]

            if not selected_player.empty:
                closest_id = selected_player["closest"].values[0]
                defender_indices = candidate_df[
                    (candidate_df["frame"] == frame)
                    & (candidate_df["id"] == closest_id)
                ].index
                candidate_df.loc[defender_indices, "selected_def"] = True

        return candidate_df


# Scenario generation classes
class ScenarioGenerator:
    """Generates counterfactual scenarios by shifting movement timing.

    This class creates alternative timeline scenarios by temporally shifting
    a selected player movement either earlier or later in time, allowing
    for analysis of timing effects on game outcomes.

    Attributes:
        selected (pd.DataFrame): The selected movement candidate DataFrame.
        scenarios (Dict[int, pd.DataFrame]): Generated scenarios indexed by shift value.
    """

    def __init__(self, selected_df):
        """Initialize with selected candidate DataFrame.

        Args:
            selected_df (pd.DataFrame): Selected movement candidate DataFrame
                containing player tracking data with movement annotations.
        """
        self.selected = selected_df
        self.scenarios = {}

    def generate_scenarios(self, shifts=range(-15, 16)):
        """Generate scenarios with temporal shifts.

        Creates multiple alternative scenarios by shifting the selected movement
        timing by different amounts, both forward and backward in time.

        Args:
            shifts (range): Range of temporal shifts to apply in frames.
                Negative values shift movement earlier, positive values later.
                Defaults to range(-15, 16) for ±15 frame shifts.

        Returns:
            Dict[int, pd.DataFrame]: Dictionary of scenario DataFrames indexed
                by shift amount.

        Note:
            - Shift=0 represents the original unmodified scenario
            - Negative shifts move movement earlier in time
            - Positive shifts move movement later in time
            - Disc positions are automatically adjusted after generation
        """
        for shift in shifts:
            if shift == 0:
                scenario_df = self.selected.copy()
            elif shift < 0:
                scenario_df = self._shift_forward(shift)
            elif shift > 0:
                scenario_df = self._shift_backward(shift)

            if scenario_df is not None:
                self.scenarios[shift] = scenario_df

        self._adjust_disc_positions()
        return self.scenarios

    def _shift_forward(self, shift):
        """Shift movement forward in time (earlier start).

        Args:
            shift (int): Negative value for forward shift

        Returns:
            pd.DataFrame or None: Shifted scenario or None if impossible
        """
        assert shift < 0

        df_shifted = self.selected.copy()
        t0 = df_shifted[df_shifted["selected"]]["frame"].min()
        max_frame = df_shifted["frame"].max()
        new_t0 = t0 + shift
        new_max_frame = max_frame + shift

        # Check if shift is possible
        if t0 - df_shifted["frame"].min() < abs(shift):
            return None

        for class_type in ["offense", "defense"]:
            player_id = self._get_player_id_for_class(df_shifted, class_type)
            if player_id is None:
                continue

            # Calculate position adjustment
            delta_x, delta_y = self._calculate_position_delta(
                df_shifted, player_id, new_t0, t0
            )

            # Remove overlapping frames
            df_shifted = self._remove_overlapping_frames(
                df_shifted, player_id, new_t0, t0
            )

            # Shift frames and adjust positions
            df_shifted = self._apply_forward_shift(
                df_shifted, player_id, t0, shift, delta_x, delta_y
            )

            # Add extrapolated frames
            df_shifted = self._add_extrapolated_frames_forward(
                df_shifted, player_id, new_max_frame, abs(shift), class_type
            )

        return df_shifted.sort_values(["frame", "id"]).reset_index(drop=True)

    def _shift_backward(self, shift):
        """Shift movement backward in time (later start).

        Args:
            shift (int): Positive value for backward shift

        Returns:
            pd.DataFrame or None: Shifted scenario or None if impossible
        """
        assert shift > 0

        df_shifted = self.selected.copy()
        t0 = df_shifted[df_shifted["selected"]]["frame"].min()
        t1 = df_shifted[df_shifted["selected"]]["frame"].max()
        max_frame = df_shifted["frame"].max()

        # Check if shift is possible
        if max_frame - t1 < shift:
            return None

        for class_type in ["offense", "defense"]:
            player_id = self._get_player_id_for_class(df_shifted, class_type)
            if player_id is None:
                continue

            # Calculate velocity for extrapolation
            vx_mean, vy_mean, v_angle = self._calculate_pre_movement_velocity(
                df_shifted, player_id, t0, shift
            )

            # Calculate position adjustment
            delta_x = vx_mean / 15 * shift
            delta_y = vy_mean / 15 * shift

            # Remove frames that will be beyond new max
            df_shifted = df_shifted[
                ~(
                    (df_shifted["id"] == player_id)
                    & (df_shifted["frame"] > max_frame - shift)
                )
            ]

            # Shift frames and adjust positions
            df_shifted = self._apply_backward_shift(
                df_shifted, player_id, t0, shift, delta_x, delta_y
            )

            # Add pre-movement frames
            df_shifted = self._add_pre_movement_frames(
                df_shifted, player_id, t0, shift, vx_mean, vy_mean, v_angle, class_type
            )

        return df_shifted.sort_values(["frame", "id"]).reset_index(drop=True)

    def _get_player_id_for_class(self, df, class_type):
        """Get player ID for the specified class."""
        column = "selected" if class_type == "offense" else "selected_def"
        selected_players = df[df[column]]["id"].values
        return int(selected_players[0]) if len(selected_players) > 0 else None

    def _calculate_position_delta(self, df, player_id, new_t0, t0):
        """Calculate position delta for forward shift."""
        new_pos = df[(df["id"] == player_id) & (df["frame"] == new_t0)][
            ["x", "y"]
        ].values[0]
        old_pos = df[(df["id"] == player_id) & (df["frame"] == t0)][["x", "y"]].values[
            0
        ]
        return new_pos[0] - old_pos[0], new_pos[1] - old_pos[1]

    def _remove_overlapping_frames(self, df, player_id, new_t0, t0):
        """Remove frames that would overlap after shifting."""
        return df[
            ~((df["id"] == player_id) & (df["frame"] >= new_t0) & (df["frame"] < t0))
        ]

    def _apply_forward_shift(self, df, player_id, t0, shift, delta_x, delta_y):
        """Apply forward shift to frames and positions."""
        # Shift frames
        mask = (df["id"] == player_id) & (df["frame"] >= t0)
        df.loc[mask, "frame"] += shift

        # Adjust positions
        shifted_t0 = t0 + shift
        pos_mask = (df["id"] == player_id) & (df["frame"] >= shifted_t0)
        df.loc[pos_mask, "x"] += delta_x
        df.loc[pos_mask, "y"] += delta_y

        return df

    def _apply_backward_shift(self, df, player_id, t0, shift, delta_x, delta_y):
        """Apply backward shift to frames and positions."""
        # Shift frames
        mask = (df["id"] == player_id) & (df["frame"] >= t0)
        df.loc[mask, "frame"] += shift

        # Adjust positions
        shifted_t0 = t0 + shift
        pos_mask = (df["id"] == player_id) & (df["frame"] >= shifted_t0)
        df.loc[pos_mask, "x"] += delta_x
        df.loc[pos_mask, "y"] += delta_y

        return df

    def _calculate_pre_movement_velocity(self, df, player_id, t0, shift):
        """Calculate average velocity before movement for extrapolation."""
        velocity_data = df[
            (df["id"] == player_id) & (df["frame"] < t0) & (df["frame"] >= t0 - shift)
        ]

        vx_mean = velocity_data["vx"].mean().round(2)
        vy_mean = velocity_data["vy"].mean().round(2)
        v_angle = np.degrees(np.arctan2(vy_mean, vx_mean)).round(2)

        return vx_mean, vy_mean, v_angle

    def _add_extrapolated_frames_forward(
        self, df, player_id, new_max_frame, num_frames, class_type
    ):
        """Add extrapolated frames for forward shift."""
        # Calculate average velocity for extrapolation
        velocity_data = df[(df["id"] == player_id) & (df["frame"] > new_max_frame - 15)]

        vx_mean = velocity_data["vx"].mean().round(2)
        vy_mean = velocity_data["vy"].mean().round(2)
        v_angle = np.degrees(np.arctan2(vy_mean, vx_mean)).round(2)

        # Get reference position and closest defender
        ref_data = df[(df["id"] == player_id) & (df["frame"] == new_max_frame)]
        ref_x, ref_y = ref_data[["x", "y"]].values[0]
        closest = ref_data["closest"].values[0]

        # Add new frames
        new_rows = []
        for i in range(1, num_frames + 1):
            x = (ref_x + vx_mean / 15 * i).round(2)
            y = (ref_y + vy_mean / 15 * i).round(2)

            new_row = [
                player_id,
                new_max_frame + i,
                class_type,
                x,
                y,
                vx_mean,
                vy_mean,
                closest,
                False,
                v_angle,
                False,
                False,
            ]
            new_rows.append(new_row)

        if new_rows:
            new_df = pd.DataFrame(new_rows, columns=df.columns)
            df = pd.concat([df, new_df], ignore_index=True)

        return df

    def _add_pre_movement_frames(
        self, df, player_id, t0, shift, vx_mean, vy_mean, v_angle, class_type
    ):
        """Add pre-movement frames for backward shift."""
        # Get reference data
        ref_data = df[(df["id"] == player_id) & (df["frame"] == t0 - 1)]
        ref_x, ref_y = ref_data[["x", "y"]].values[0]
        closest = df[df["id"] == player_id]["closest"].values[0]

        # Add new frames
        new_rows = []
        for i in range(1, shift + 1):
            x = (ref_x + vx_mean / 15 * i).round(2)
            y = (ref_y + vy_mean / 15 * i).round(2)

            new_row = [
                player_id,
                t0 - 1 + i,
                class_type,
                x,
                y,
                vx_mean,
                vy_mean,
                closest,
                False,
                v_angle,
                False,
                False,
            ]
            new_rows.append(new_row)

        if new_rows:
            new_df = pd.DataFrame(new_rows, columns=df.columns)
            df = pd.concat([df, new_df], ignore_index=True)

        return df

    def _adjust_disc_positions(self):
        """Adjust disc positions to maintain consistency with ball possession."""
        for shift, scenario_df in self.scenarios.items():
            scenario_df = self._resolve_disc_possession(scenario_df)
            scenario_df = self._interpolate_disc_positions(scenario_df)
            self.scenarios[shift] = scenario_df

    def _resolve_disc_possession(self, scenario_df):
        """Resolve disc possession conflicts and ensure consistency."""
        scenario_df = scenario_df.sort_values(by="frame", ascending=False).reset_index(
            drop=True
        )
        selected_id = int(scenario_df[scenario_df["selected"]]["id"].values[0])

        current_holder = None
        for frame, frame_data in scenario_df.groupby("frame"):
            holders = frame_data[frame_data["holder"]]

            if len(holders) == 1:
                current_holder = holders["id"].values[0]
            elif len(holders) >= 2:
                # Resolve conflicts - prioritize selected player
                scenario_df.loc[
                    (scenario_df["frame"] == frame)
                    & (scenario_df["id"] != selected_id),
                    "holder",
                ] = False
                current_holder = selected_id
            else:
                # No holder - assign to current holder if available
                if current_holder is not None:
                    scenario_df.loc[
                        (scenario_df["frame"] == frame)
                        & (scenario_df["id"] == current_holder),
                        "holder",
                    ] = True

        return scenario_df.sort_values(by=["frame", "id"]).reset_index(drop=True)

    def _interpolate_disc_positions(self, scenario_df):
        """Interpolate disc positions based on holder positions."""
        # Clear disc positions except for the last frame
        max_frame = scenario_df["frame"].max()
        disc_mask = (scenario_df["class"] == "disc") & (
            scenario_df["frame"] != max_frame
        )
        scenario_df.loc[disc_mask, ["x", "y"]] = None

        # Set disc position to holder's position
        holder_frames = scenario_df[scenario_df["holder"]]["frame"].unique()
        for frame in holder_frames:
            holder_pos = scenario_df[
                (scenario_df["holder"]) & (scenario_df["frame"] == frame)
            ][["x", "y"]].values[0]

            scenario_df.loc[
                (scenario_df["class"] == "disc") & (scenario_df["frame"] == frame),
                ["x", "y"],
            ] = holder_pos

        # Interpolate missing positions
        disc_data = scenario_df[scenario_df["class"] == "disc"].copy()
        disc_data[["x", "y"]] = (
            disc_data[["x", "y"]]
            .interpolate(method="linear", limit_direction="both")
            .round(2)
        )

        scenario_df.loc[scenario_df["class"] == "disc", ["x", "y"]] = disc_data[
            ["x", "y"]
        ].values

        return scenario_df


# Evaluation classes
class VTCSEvaluator:
    """Evaluates VTCS scenarios using frame-wise and scenario-wise metrics.

    This class computes strategic value metrics for different timing scenarios
    using weighted Ultimate Probabilistic Player Control Fields (wUPPCF).

    Attributes:
        v_frame (Dict[int, List[float]]): Frame-wise values for each scenario.
        v_scenario (Dict[int, float]): Scenario-level values for each shift.
        v_timing (float): Timing effectiveness metric.
    """

    def __init__(self):
        """Initialize evaluator with empty metric storage."""
        self.v_frame = {}
        self.v_scenario = {}
        self.v_timing = None

    def evaluate_scenarios(self, scenarios, file_name, candidate_id):
        """Evaluate all scenarios and calculate metrics.

        Computes frame-wise and scenario-wise strategic value metrics for
        each generated scenario using pre-computed wUPPCF data.

        Args:
            scenarios (Dict[int, pd.DataFrame]): Dictionary of scenario DataFrames
                indexed by shift amount.
            file_name (str): Input file name for wUPPCF file lookup.
            candidate_id (str): Candidate identifier for file naming.

        Returns:
            Dict[str, Any]: Evaluation results containing:
                - v_frame: Frame-wise values for each scenario
                - v_scenario: Scenario-level summary values
                - v_timing: Timing effectiveness metric
                - best_timing: Optimal shift value

        Note:
            Requires pre-computed wUPPCF files in data/input/player_wUPPCF/
            directory with naming pattern: {file_name}-{candidate_id}_{shift}.npy
        """
        self.v_frame = {}
        self.v_scenario = {}

        # Evaluate each scenario
        for shift, scenario_df in scenarios.items():
            wuppcf_file = self._get_wuppcf_file_path(file_name, candidate_id, shift)

            if not os.path.exists(wuppcf_file):
                continue

            wuppcf = np.load(wuppcf_file)
            v_frame_list = self._calculate_v_frame_series(scenario_df, wuppcf)
            v_scenario_val = self._calculate_v_scenario(v_frame_list)

            self.v_frame[shift] = v_frame_list
            self.v_scenario[shift] = v_scenario_val

        # Calculate V_timing
        self._calculate_v_timing()

        return self._create_results_dict()

    def _get_wuppcf_file_path(self, file_name, candidate_id, shift):
        """Generate wUPPCF file path for given parameters."""
        formatted_id = candidate_id.replace("-", "_")
        return f"data/input/player_wUPPCF/{file_name}-{formatted_id}_{shift}.npy"

    def _calculate_v_frame_series(
        self,
        scenario_df,
        wuppcf,
        xgrid=np.arange(1, 94, 2),
        ygrid=np.arange(37 / 19 / 2, 37, 37 / 19),
    ):
        """Calculate V_frame series for all frames in scenario.

        Args:
            scenario_df (pd.DataFrame): Scenario data
            wuppcf (np.ndarray): Pre-computed wUPPCF values
            xgrid (np.ndarray): X-coordinate grid
            ygrid (np.ndarray): Y-coordinate grid

        Returns:
            list: V_frame values for each frame
        """
        unique_frames = sorted(scenario_df["frame"].unique())
        v_frame_list = []

        for frame in unique_frames:
            frame_df = scenario_df[scenario_df["frame"] == frame]

            # Get offensive players (excluding holder)
            off_df = (
                frame_df[(frame_df["class"] == "offense") & (~frame_df["holder"])]
                .sort_values(by="id")
                .reset_index(drop=True)
            )

            # Get disc position
            disc_pos = frame_df[frame_df["class"] == "disc"][["x", "y"]].values[0]

            # Find selected player
            selected_players = frame_df[frame_df["selected"]]["id"].values
            if len(selected_players) == 0:
                continue

            selected_id = selected_players[0]
            off_ids = off_df["id"].values

            # Find index of selected player in offensive players array
            selected_indices = np.where(off_ids == selected_id)[0]
            if len(selected_indices) == 0:
                continue

            selected_index = selected_indices[0]

            # Calculate V_frame for this frame
            v_frame = self._calculate_single_v_frame(
                wuppcf[selected_index, frame - min(unique_frames) - 1, :, :],
                off_df.iloc[selected_index][["x", "y", "vx", "vy"]],
                xgrid,
                ygrid,
                disc_pos,
            )

            v_frame_list.append(v_frame)

        return v_frame_list

    def _calculate_single_v_frame(
        self, wuppcf_slice, player_data, xgrid, ygrid, disc_pos, disc_speed=10
    ):
        """Calculate V_frame value for a single frame and player.

        Args:
            wuppcf_slice (np.ndarray): wUPPCF values for this player/frame
            player_data (pd.Series): Player position and velocity data
            xgrid (np.ndarray): X-coordinate grid
            ygrid (np.ndarray): Y-coordinate grid
            disc_pos (tuple): Disc position (x, y)
            disc_speed (float): Disc movement speed

        Returns:
            float: V_frame value
        """
        x_p, y_p, vx, vy = player_data
        x_d, y_d = disc_pos

        # Calculate interception geometry
        A = vx**2 + vy**2 - disc_speed**2
        B = 2 * (vx * (x_p - x_d) + vy * (y_p - y_d))
        C = (x_p - x_d) ** 2 + (y_p - y_d) ** 2

        discriminant = B**2 - 4 * A * C

        if discriminant < 0:
            return 0.0

        # Calculate intersection time
        t1 = (-B + np.sqrt(discriminant)) / (2 * A)
        t2 = (-B - np.sqrt(discriminant)) / (2 * A)
        t = max(t1, t2)

        if t <= 0:
            return 0.0

        # Calculate target position and uncertainty radius
        x_t = x_p + vx * t
        y_t = y_p + vy * t
        radius = np.sqrt((vx * t * 0.5) ** 2 + (vy * t * 0.5) ** 2)

        # Create spatial mask for value aggregation
        X, Y = np.meshgrid(xgrid, ygrid)
        mask = (X - x_t) ** 2 + (Y - y_t) ** 2 <= radius**2

        if not np.any(mask):
            return 0.0

        # Calculate weighted average value
        v_frame = np.mean(np.flipud(wuppcf_slice)[mask]).round(3)
        return v_frame

    def _calculate_v_scenario(self, v_frame_list):
        """Calculate V_scenario from V_frame series using moving average.

        Args:
            v_frame_list (list): List of V_frame values

        Returns:
            float: V_scenario value (maximum of smoothed series)
        """
        if not v_frame_list:
            return 0.0

        # Apply moving average filter and take maximum
        smoothed = np.convolve(v_frame_list, np.ones(15) / 15, mode="same")
        return smoothed.max().round(3)

    def _calculate_v_timing(self):
        """Calculate V_timing as difference between actual and best scenario."""
        if not self.v_scenario:
            self.v_timing = None
            return

        v_actual = self.v_scenario.get(0, None)

        # Get best scenario value (excluding actual)
        other_scenarios = {k: v for k, v in self.v_scenario.items() if k != 0}
        v_best = max(other_scenarios.values()) if other_scenarios else None

        if v_actual is not None and v_best is not None:
            self.v_timing = v_actual - v_best
        else:
            self.v_timing = None

    def _create_results_dict(self):
        """Create results dictionary with all evaluation metrics."""
        best_shift = None
        if self.v_scenario:
            # Find the key with the maximum value
            max_value = max(self.v_scenario.values())
            for key, value in self.v_scenario.items():
                if value == max_value:
                    best_shift = key
                    break

        return {
            "v_frame": self.v_frame,
            "v_scenario": self.v_scenario,
            "v_timing": self.v_timing,
            "best_timing": best_shift,
        }


class VTCS:
    """Valuing Timing by Counterfactual Scenarios (VTCS) analysis system.

    A class for analyzing ultimate frisbee player movements and generating
    counterfactual scenarios to evaluate timing decisions.

    Attributes:
        play (pd.DataFrame): Processed tracking data with derived columns.
        movement_detector (MovementDetector): Component for detecting player movements.
        scenario_generator (ScenarioGenerator): Component for generating scenarios.
        evaluator (VTCSEvaluator): Component for evaluating scenarios.
        candidates (Dict[str, pd.DataFrame]): Dictionary of detected movement candidates.
        selected (pd.DataFrame): Currently selected candidate for analysis.
        scenarios (Dict[int, pd.DataFrame]): Generated scenarios with different time shifts.
    """

    def __init__(self, ultimate_track_df):
        """Initialize VTCS with ultimate frisbee tracking data.

        Args:
            ultimate_track_df (pd.DataFrame): DataFrame containing tracking data with columns:
                - 'id': Player/disc ID
                - 'frame': Frame number
                - 'x', 'y': Position coordinates
                - 'vx', 'vy': Velocity components
                - 'ax', 'ay': Acceleration components
                - 'class': Player type ('offense', 'defense', 'disc')
                - 'holder': Boolean indicating if player holds the disc
                - 'closest': ID of closest opposing player
        """
        # Add derived columns to the play data
        self.play = add_derived_columns(ultimate_track_df)

        # Initialize components
        self.movement_detector = MovementDetector(self.play)
        self.scenario_generator = None
        self.evaluator = VTCSEvaluator()

        # Storage for analysis results
        self.candidates = {}
        self.selected = None
        self.scenarios = {}

    def detect_candidates(self):
        """Detect movement candidates from the play DataFrame.

        This method performs a three-step process to identify potential movement
        candidates:

        1. Detect movement initiation based on acceleration and velocity criteria
        2. Deselect movements based on proximity and direction constraints
        3. Extract candidate DataFrames with temporal windows

        The method updates the internal candidates dictionary with detected movements.
        Each candidate is identified by a unique key in the format "{player_id}-{sequence_number}".

        Note:
            This method modifies the internal state of the VTCS object by populating
            the candidates dictionary.
        """
        # Update play data with detection results
        self.movement_detector.detect_movement_initiation()
        self.play = self.movement_detector.play

        # Filter out crowded movements
        self.movement_detector.deselect_crowded_movements()
        self.play = self.movement_detector.play

        # Extract candidates
        self.candidates = self.movement_detector.extract_candidates()

    def generate_scenarios(self, shifts=range(-15, 16)):
        """Generate counterfactual scenarios by shifting selected movement timing.

        Creates alternative scenarios by temporally shifting the selected movement
        either forward (negative shifts) or backward (positive shifts) in time.

        Args:
            shifts (range, optional): Range of temporal shifts to apply in frames.
                Negative values shift movement earlier, positive values shift later.
                Defaults to range(-15, 16) for ±15 frame shifts.

        Raises:
            ValueError: If no candidate has been selected for analysis.

        Note:
            After generating all scenarios, disc positions are automatically
            adjusted to maintain consistency with ball possession.
        """
        if self.selected is None:
            raise ValueError("No candidate selected. Call select_candidate() first.")

        self.scenario_generator = ScenarioGenerator(self.selected)
        self.scenarios = self.scenario_generator.generate_scenarios(shifts)

    def select_candidate(self, candidate_id):
        """Select a candidate for scenario generation.

        Args:
            candidate_id (str): ID of the candidate to select. Must exist in the
                candidates dictionary.

        Raises:
            ValueError: If the specified candidate_id is not found in the candidates
                dictionary.
        """
        if candidate_id not in self.candidates:
            raise ValueError(f"Candidate {candidate_id} not found.")

        self.selected = self.candidates[candidate_id]

    def evaluate(self, file_name: str, candidate_id: str):
        """Evaluate all generated scenarios using frame-wise and scenario-wise metrics.

        Calculates evaluation metrics for each scenario including V_frame (frame-wise
        values), V_scenario (scenario summary), and V_timing (timing effectiveness).

        Args:
            file_name (str): Name of the input file used for evaluation, primarily
                for logging and result identification.
            candidate_id (str): Identifier for the selected candidate movement.

        Returns:
            Dict[str, Any]: Dictionary containing evaluation results with keys:
                - 'v_frame': Dict mapping shift -> list of frame-wise values
                - 'v_scenario': Dict mapping shift -> scenario summary value
                - 'v_timing': Float representing timing effectiveness
                - 'best_timing': Int representing the optimal shift value

        Raises:
            ValueError: If no scenarios have been generated prior to evaluation.

        Note:
            V_timing is calculated as the difference between actual (shift=0)
            and best possible scenario performance. Requires pre-computed wUPPCF
            files to be available in the data/input/player_wUPPCF/ directory.
        """
        if not self.scenarios:
            raise ValueError("No scenarios generated. Call generate_scenarios() first.")

        return self.evaluator.evaluate_scenarios(
            self.scenarios, file_name, candidate_id
        )


def main():
    """Main function to demonstrate VTCS analysis workflow.

    Loads ultimate frisbee tracking data, detects movement candidates,
    allows user selection of a candidate, generates counterfactual scenarios,
    and evaluates the timing values.

    The workflow includes:
        1. Load tracking data from CSV
        2. Detect movement candidates
        3. Interactive candidate selection
        4. Generate temporal shift scenarios
        5. Evaluate scenario performance

    Example:
        Run the script directly to start the interactive analysis::

            $ python vtcs.py

    Note:
        Requires input data file at 'data/input/UltimateTrack/1_1_2.csv'.
        Evaluation requires corresponding wUPPCF files in 'data/input/player_wUPPCF/'.
    """
    pd.set_option("display.max_rows", None)
    # Load data and initialize VTCS
    file_name = "data/input/UltimateTrack/1_1_2.csv"
    ultimate_track_df = pd.read_csv(file_name)
    vtcs = VTCS(ultimate_track_df)

    # Detect movement candidates
    vtcs.detect_candidates()

    if vtcs.candidates:
        # Interactive candidate selection
        while True:
            print("Available candidates:", ", ".join(vtcs.candidates.keys()))
            candidate_index = input("Enter candidate index to select (e.g., '1-1'): ")

            if candidate_index == "q":
                print("Exiting.")
                sys.exit(0)

            if candidate_index in vtcs.candidates:
                vtcs.select_candidate(candidate_index)
                if vtcs.selected is not None and hasattr(vtcs.selected, "attrs"):
                    selected_attrs = vtcs.selected.attrs
                    print(
                        f"Selected candidate: {selected_attrs['movement_player_id']} "
                        f"from frame {selected_attrs['movement_start_frame']} "
                        f"to {selected_attrs['movement_end_frame']}"
                    )
                else:
                    print(f"Selected candidate: {candidate_index}")
                break
            else:
                print("Invalid candidate index. Please try again.")

        # Generate scenarios and evaluate
        vtcs.generate_scenarios()

        evaluation_results = vtcs.evaluate(
            file_name.split("/")[-1].split(".")[0], candidate_index
        )

        # Display results
        print("\nEvaluation Results:")
        print("V_scenario:", evaluation_results["v_scenario"])
        print("V_timing:", evaluation_results["v_timing"])
        print("Best timing shift:", evaluation_results["best_timing"])


if __name__ == "__main__":
    main()
