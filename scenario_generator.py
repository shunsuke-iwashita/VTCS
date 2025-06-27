"""Scenario generation for VTCS analysis."""

import numpy as np
import pandas as pd


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
