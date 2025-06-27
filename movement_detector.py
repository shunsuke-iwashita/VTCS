"""Movement detection and candidate extraction for VTCS analysis."""

import itertools

import numpy as np

from data_utils import (
    circular_average,
    is_within_forward_direction,
    update_selected_and_length,
)


class MovementDetector:
    """Detects and extracts movement candidates from tracking data."""

    def __init__(self, play_df):
        """Initialize with play DataFrame.

        Args:
            play_df (pd.DataFrame): Tracking data with derived columns
        """
        self.play = play_df.copy()

    def detect_movement_initiation(self, v_threshold=3.0, a_threshold=4.0):
        """Detect movement initiation and perform temporal expansion.

        Args:
            v_threshold (float): Minimum velocity threshold for movement detection in m/s
            a_threshold (float): Minimum acceleration threshold for movement initiation in m/s²
        """
        self._detect_initial_movements(v_threshold, a_threshold)
        self._expand_forward(v_threshold)
        self._filter_by_length()
        self._expand_backward()
        print(self.play[self.play["selected"]].sort_values(["id", "frame"]))
        print(self.play.columns)
        self._cleanup_columns()

    def _detect_initial_movements(self, v_threshold, a_threshold):
        """Detect initial movement based on acceleration and velocity criteria."""
        self.play["selected"] = False
        self.play["prev_holder"] = self.play.groupby("id")["holder"].shift(30)

        # Calculate angle difference between velocity and acceleration
        angle_diff = np.abs(
            (self.play["v_angle"] - self.play["a_angle"] + 180) % 360 - 180
        )

        # Define selection criteria
        selection_criteria = (
            (self.play["frame"] > 1)
            & (self.play["class"] == "offense")
            & (~self.play["holder"])
            & (self.play["prev_holder"] != True)
            & (self.play["a_mag"] > a_threshold)
            & (angle_diff < 90)
        )

        self.play.loc[selection_criteria, "selected"] = True

    def _expand_forward(self, v_threshold):
        """Expand selection forward in time based on velocity consistency."""
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
        """Filter selections based on sequence length."""
        self.play = self.play.groupby("id", group_keys=False).apply(
            update_selected_and_length
        )

    def _expand_backward(self):
        """Expand selection backward in time based on velocity magnitude."""
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
        """Remove temporary calculation columns."""
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

        Args:
            distance_threshold (float): Distance threshold for crowded conditions
            player_threshold (int): Minimum number of nearby players for crowded condition
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
        """Check if movement should be deselected due to crowded conditions."""
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

        Args:
            window (int): Number of frames to include before and after movement

        Returns:
            dict: Dictionary of candidate DataFrames
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
        """Create a candidate DataFrame for a movement sequence."""
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
        """Mark closest defenders during movement period."""
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
