"""Evaluation system for VTCS scenarios."""

import os

import numpy as np


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
