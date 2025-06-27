import itertools
import os
import sys

import numpy as np
import pandas as pd


class VTCS:
    """Vision-based Tactical Counterfactual Scenarios (VTCS) analysis system.

    A class for analyzing ultimate frisbee player movements and generating
    counterfactual scenarios to evaluate timing decisions.
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
        self.play = ultimate_track_df
        self.candidates = {}  # dict: {id: DataFrame}
        self.selected = None  # DataFrame
        self.scenarios = {}  # dict: {shift: DataFrame of scenario}

        # --- 評価指標（初期化時はNoneや空で持つ） ---
        self.v_frame = {}  # dict: {shift: list of values}
        self.v_scenario = {}  # dict: {shift: value}
        self.v_timing = None  # float

        # 必要なら
        self.optimal_shift = None
        self.result_summary = None

        self.play["v_mag"] = np.sqrt(self.play["vx"] ** 2 + self.play["vy"] ** 2)
        self.play["a_mag"] = np.sqrt(self.play["ax"] ** 2 + self.play["ay"] ** 2)
        self.play["v_angle"] = (
            np.arctan2(self.play["vy"], self.play["vx"]) * 180 / np.pi
        )
        self.play["a_angle"] = (
            np.arctan2(self.play["ay"], self.play["ax"]) * 180 / np.pi
        )
        self.play["diff_v_angle"] = (
            self.play.groupby("id")["v_angle"]
            .diff()
            .abs()
            .apply(lambda x: min(x, 360 - x) if pd.notnull(x) else x)
        )

    def detect_candidates(self):
        """Detect movement candidates from the play DataFrame.

        This method performs a three-step process to identify potential movement
        candidates:
        1. Detect movement initiation based on acceleration and velocity criteria
        2. Deselect movements based on proximity and direction constraints
        3. Extract candidate DataFrames with temporal windows
        """
        # Detect movement initiation from the play DataFrame.
        self.detect_movement_initiation()  # 例: play DataFrameを更新する関数
        self.deselect_movement_initiation()  # 検出された動き出しを除外する関数
        self.extract_movement_candidates()  # 例: [df1, df2, ...]

    def generate_scenarios(self, shifts=range(-15, 16)):
        """Generate counterfactual scenarios by shifting selected movement timing.

        Creates alternative scenarios by temporally shifting the selected movement
        either forward (negative shifts) or backward (positive shifts) in time.

        Args:
            shifts (range, optional): Range of temporal shifts to apply in frames.
                Negative values shift movement earlier, positive values shift later.
                Defaults to range(-15, 16) for ±15 frame shifts.

        Note:
            After generating all scenarios, disc positions are automatically
            adjusted to maintain consistency with ball possession.
        """
        for shift in shifts:
            if shift == 0:
                scenario_df = self.selected.copy()
            elif shift < 0:
                scenario_df = self.shift_forward(
                    shift
                )  # 例: DataFrameを前方にシフトする関数
            elif shift > 0:
                scenario_df = self.shift_backward(
                    shift
                )  # 例: DataFrameを後方にシフトする関数
            self.scenarios[shift] = scenario_df

        self.adjust_disc_positions()

    def evaluate(self, file_name: str, candidate_id: str):
        """Evaluate all generated scenarios using frame-wise and scenario-wise metrics.

        Calculates evaluation metrics for each scenario including V_frame (frame-wise
        values), V_scenario (scenario summary), and V_timing (timing effectiveness).

        Args:
            file_name (str): Name of the input file used for evaluation, primarily
                for logging and result identification.
            candidate_id (str): Identifier for the selected candidate movement.

        Returns:
            dict: Dictionary containing evaluation results with keys:
                - 'v_frame': Dict mapping shift -> list of frame-wise values
                - 'v_scenario': Dict mapping shift -> scenario summary value
                - 'v_timing': Float representing timing effectiveness
                - 'best_timing': Int representing the optimal shift value

        Note:
            V_timing is calculated as the difference between actual (shift=0)
            and best possible scenario performance.

        TODO:
            - Implement calc_wuppcf method to compute wUPPCF values dynamically
        """
        self.v_frame = {}
        self.v_scenario = {}

        for shift, scenario_df in self.scenarios.items():
            wuppcf_file = f"data/input/player_wUPPCF/{file_name}-{candidate_id.replace('-', '_')}_{shift}.npy"
            if not os.path.exists(wuppcf_file):
                continue
            wuppcf = np.load(wuppcf_file)
            # wuppcf = self.calc_wuppcf(scenario_df)
            v_frame_list = self.get_v_frame_series(
                scenario_df, wuppcf
            )  # 例: [float, ...]を返す
            v_scenario_val = self.calc_v_scenario(v_frame_list)  # 例: floatを返す

            self.v_frame[shift] = v_frame_list
            self.v_scenario[shift] = v_scenario_val

        # V_timing計算
        v_actual = self.v_scenario.get(0, None)
        v_best = max([v for k, v in self.v_scenario.items() if k != 0])
        best_shift = max(self.v_scenario, key=self.v_scenario.get)
        self.v_timing = (
            v_actual - v_best if v_actual is not None and v_best is not None else None
        )

        return {
            "v_frame": self.v_frame,  # {shift: [v1, v2, ...]}
            "v_scenario": self.v_scenario,  # {shift: value}
            "v_timing": self.v_timing,  # float
            "best_timing": best_shift,  # 最適なシフト
        }

    def detect_movement_initiation(
        self, v_threshold: float = 3.0, a_threshold: float = 4.0
    ):
        """Detect movement initiation and perform temporal expansion.

        Identifies when offensive players begin significant movements based on
        acceleration and velocity criteria, then expands the detection both
        forward and backward in time to capture complete movement sequences.

        Args:
            v_threshold (float, optional): Minimum velocity threshold for movement
                detection in m/s. Defaults to 3.0.
            a_threshold (float, optional): Minimum acceleration threshold for movement
                initiation in m/s². Defaults to 4.0.

        Note:
            This method modifies self.play by adding a 'selected' column and
            removes temporary calculation columns after processing.
        """
        # ---------- detect movement initiation ----------
        self.play["selected"] = False
        self.play["prev_holder"] = self.play.groupby("id")["holder"].shift(30)
        angle_diff = np.abs(
            (self.play["v_angle"] - self.play["a_angle"] + 180) % 360 - 180
        )
        # Define the selection criteria
        selection_criteria = (
            (self.play["frame"] > 1)
            & (self.play["class"] == "offense")
            & (~self.play["holder"])
            & (self.play["prev_holder"] != True)
            & (self.play["a_mag"] > a_threshold)
            & (angle_diff < 90)
        )
        self.play.loc[selection_criteria, "selected"] = True

        # ---------- forward expansion ----------
        def circular_average(angles):
            radians = np.deg2rad(angles)
            sin_average = np.mean(np.sin(radians))
            cos_average = np.mean(np.cos(radians))
            average_radian = np.arctan2(sin_average, cos_average)
            average_angle = np.rad2deg(average_radian)
            average_angle = average_angle % 360
            return average_angle

        for id, id_df in self.play.groupby("id"):
            id_df = id_df.sort_values("frame").reset_index()
            frames = id_df["frame"].values
            v_angles = id_df["v_angle"].values
            selected = id_df["selected"].values
            diff_v_angle = id_df["diff_v_angle"].values
            v_mag = id_df["v_mag"].values
            holder = id_df["holder"].values

            v_angle_history = []
            for i in range(len(frames)):
                frame = frames[i]
                if not selected[i]:
                    v_angle_history = []
                    continue
                v_angle_history.append(v_angles[i])

                next_idx = np.where(frames == frame + 1)[0]
                if len(next_idx) == 0:
                    continue
                ni = next_idx[0]

                angle_diff = abs((v_angles[ni] - circular_average(v_angle_history)))
                angle_diff = min(angle_diff, 360 - angle_diff)

                # Condition for selecting the next frame
                cond = (
                    (diff_v_angle[ni] <= 20)
                    and (v_mag[ni] > v_threshold)
                    and (not holder[ni])
                    and (angle_diff <= 90)
                )
                selected[ni] = cond

            self.play.loc[id_df["index"], "selected"] = selected

        # ---------- deselect candidates based on length ----------
        def update_selected_and_length(group):
            group = group.sort_values("frame").reset_index(drop=True)
            selected = group["selected"].values
            frames = group["frame"].values

            diff = np.diff(selected.astype(int), prepend=0)
            starts = np.where((diff == 1))[0]
            ends = np.where((diff == -1))[0] - 1

            if selected[-1]:
                if len(ends) < len(starts):
                    ends = np.append(ends, len(selected) - 1)

            length = np.zeros(len(selected), dtype=int)
            for s, e in zip(starts, ends):
                idx = np.arange(s, e + 1)
                if np.all(np.diff(frames[idx]) == 1):
                    seq_length = e - s + 1
                    length[idx] = seq_length

            group["length"] = length
            group.loc[(group["length"] < 15) | (group["length"] > 75), "selected"] = (
                False
            )
            return group

        self.play = self.play.groupby("id", group_keys=False).apply(
            update_selected_and_length
        )

        # ---------- backward expansion ----------
        self.play.set_index(["id", "frame"], inplace=True)

        max_frame = self.play.index.get_level_values("frame").max()
        min_frame = self.play.index.get_level_values("frame").min()

        for frame in range(max_frame, min_frame, -1):
            selected_idx = self.play.index[
                self.play["selected"]
                & (self.play.index.get_level_values("frame") == frame)
            ]
            for idx in selected_idx:
                id = idx[0]
                prev_idx = (id, frame - 1)
                if prev_idx in self.play.index:
                    if (
                        not self.play.at[prev_idx, "selected"]
                        and self.play.at[prev_idx, "v_mag"] > 0.05
                        and self.play.at[prev_idx, "v_mag"] - self.play.at[idx, "v_mag"]
                        < 0.05
                    ):
                        self.play.at[prev_idx, "selected"] = True

        self.play.reset_index(inplace=True)
        self.play.drop(
            columns=[
                "ax",
                "ay",
                "v_mag",
                "a_mag",
                "a_angle",
                "diff_v_angle",
                "prev_holder",
                "length",
            ],
            inplace=True,
        )

    def deselect_movement_initiation(self):
        """Deselect movement candidates based on proximity and running direction.

        Filters out movement candidates that occur in crowded areas or when
        multiple players are moving in the same direction, as these are likely
        less significant individual movements.

        The deselection criteria include:
            - Distance threshold of 5.0 meters for determining crowded conditions
            - Player threshold of 2 or more nearby players
            - Forward direction cone of 45 degrees for detecting similar movements

        Note:
            Uses distance threshold of 5.0 meters and player threshold of 2 to
            determine crowded conditions. Also checks for players moving within
            45-degree forward direction cone.
        """

        def is_within_forward_direction(target_direction, dx, dy):
            angle = np.degrees(np.arctan2(dy, dx))
            relative_angle = (angle - target_direction + 360) % 360
            return (0 <= relative_angle) & (relative_angle <= 45) | (
                315 <= relative_angle
            ) & (relative_angle < 360)

        distance_threshold = 5.0
        player_threshold = 2

        for id_val in self.play["id"].unique():
            selected_frames = self.play.loc[
                (self.play["id"] == id_val) & (self.play["selected"]), "frame"
            ].values
            continuous_frames_list = [
                list(g)
                for _, g in itertools.groupby(
                    selected_frames, key=lambda n, c=itertools.count(): n - next(c)
                )
            ]

            if continuous_frames_list:
                for continuous_frames in continuous_frames_list:
                    last_frame = continuous_frames[-1]
                    last_frame_data = self.play[self.play["frame"] == last_frame]

                    # 対象プレイヤーのデータ
                    target_row = last_frame_data[last_frame_data["id"] == id_val]
                    if target_row.empty:
                        continue
                    tx, ty, tdir = target_row[["x", "y", "v_angle"]].values[0]

                    # 他のoffenseプレイヤーのデータ
                    others = last_frame_data[
                        (last_frame_data["id"] != id_val)
                        & (last_frame_data["class"] == "offense")
                    ]
                    if others.empty:
                        continue
                    ox = others["x"].values
                    oy = others["y"].values

                    # 距離計算（ベクトル化）
                    dx = ox - tx
                    dy = oy - ty
                    distances = np.hypot(dx, dy)
                    count = np.sum(distances <= distance_threshold)

                    # 進行方向内判定（ベクトル化）
                    direction_count = np.sum(is_within_forward_direction(tdir, dx, dy))

                    if count >= player_threshold or direction_count >= 2:
                        self.play.loc[
                            (self.play["id"] == id_val)
                            & (self.play["frame"].isin(continuous_frames)),
                            "selected",
                        ] = False

        self.play["v_angle"] = self.play["v_angle"].round(2)

    def extract_movement_candidates(self, window=30):
        """Extract movement candidates with temporal windows.

        Extracts continuous sequences of selected movements for each player,
        adding temporal context windows before and after each movement sequence.

        Args:
            window (int, optional): Number of frames to include before and after
                each movement sequence for context. Defaults to 30.

        The extraction process:
            1. Groups selected movements by player ID
            2. Identifies continuous frame sequences for each player
            3. Adds temporal context windows around each sequence
            4. Creates candidate DataFrames with movement metadata
            5. Marks closest defenders during movement periods

        Note:
            Creates candidate DataFrames with attributes:
            - movement_player_id: ID of the moving player
            - movement_start_frame: First frame of movement
            - movement_end_frame: Last frame of movement

            Also sets selected_def=True for the closest defender during movement.
        """
        play = self.play

        # 選択された全選手・全フレームを抽出
        selected = play[play["selected"]]
        # 連続区間を検出するため、各playerごとに処理
        for player_id, group in selected.groupby("id"):
            frames = group["frame"].sort_values().values
            if len(frames) == 0:
                continue
            # 連続区間ごとにグループ化
            # 差分が1でないところでグループが切れる
            split_idx = np.where(np.diff(frames) != 1)[0] + 1
            frame_groups = np.split(frames, split_idx)
            for i, frames_in_group in enumerate(frame_groups):
                start_frame = max(frames_in_group.min() - window, play["frame"].min())
                end_frame = min(frames_in_group.max() + window, play["frame"].max())
                candidate_df = play[
                    (play["frame"] >= start_frame) & (play["frame"] <= end_frame)
                ].copy()
                candidate_df["selected"] = False
                candidate_df.loc[
                    (candidate_df["id"] == player_id)
                    & (candidate_df["frame"].isin(frames_in_group)),
                    "selected",
                ] = True
                candidate_df.attrs["movement_player_id"] = player_id
                candidate_df.attrs["movement_start_frame"] = frames_in_group.min()
                candidate_df.attrs["movement_end_frame"] = frames_in_group.max()
                candidate_df["selected_def"] = False
                # 同じフレームのグループ内で、selectedがTrueのclosestの値をidにもつデータのselected_defをTrueに変更する
                for frame in frames_in_group:
                    frame_df = candidate_df[candidate_df["frame"] == frame]
                    selected_row = frame_df[frame_df["selected"]]
                    if not selected_row.empty:
                        closest_id = selected_row["closest"].values[0]
                        idx = candidate_df[
                            (candidate_df["frame"] == frame)
                            & (candidate_df["id"] == closest_id)
                        ].index
                        candidate_df.loc[idx, "selected_def"] = True
                self.candidates[f"{player_id}-{i+1}"] = candidate_df

    def shift_forward(self, shift):
        """Shift selected candidate DataFrame forward in time.

        Moves the selected movement sequence earlier in time by removing frames
        from the beginning and extrapolating new frames at the end based on
        average velocity patterns.

        Args:
            shift (int): Number of frames to shift forward (must be negative).
                Negative values move the movement earlier in time.

        Returns:
            pd.DataFrame or None: Modified DataFrame with shifted movement, or None
                if insufficient data exists for the requested shift.

        Raises:
            AssertionError: If shift is not negative (shift >= 0).

        Note:
            Processes both offense and defense players, adjusting positions and
            maintaining velocity consistency through extrapolation.
        """
        assert shift < 0
        df_shifted = self.selected.copy()
        t0 = df_shifted[df_shifted["selected"]]["frame"].min()
        max_frame = df_shifted["frame"].max()
        new_t0 = t0 + shift
        new_max_frame = max_frame + shift

        if t0 - df_shifted["frame"].min() < abs(shift):
            return None

        for class_type in ["offense", "defense"]:
            column = "selected" if class_type == "offense" else "selected_def"

            # Get the selected player id
            selected_id = int(df_shifted[df_shifted[column]]["id"].values[0])

            # Get the delta_x and delta_y
            delta_x = (
                df_shifted[
                    (df_shifted["id"] == selected_id) & (df_shifted["frame"] == new_t0)
                ]["x"].values[0]
                - df_shifted[
                    (df_shifted["id"] == selected_id) & (df_shifted["frame"] == t0)
                ]["x"].values[0]
            )
            delta_y = (
                df_shifted[
                    (df_shifted["id"] == selected_id) & (df_shifted["frame"] == new_t0)
                ]["y"].values[0]
                - df_shifted[
                    (df_shifted["id"] == selected_id) & (df_shifted["frame"] == t0)
                ]["y"].values[0]
            )

            # Remove duplicate frames by shifting the frame
            df_shifted = df_shifted[
                ~(
                    (df_shifted["id"] == selected_id)
                    & (df_shifted["frame"] >= new_t0)
                    & (df_shifted["frame"] < t0)
                )
            ]

            # Shift the frames
            df_shifted.loc[
                (df_shifted["id"] == selected_id) & (df_shifted["frame"] >= t0), "frame"
            ] += shift

            # Correct the x and y positions
            df_shifted.loc[
                (df_shifted["id"] == selected_id) & (df_shifted["frame"] >= new_t0), "x"
            ] += delta_x
            df_shifted.loc[
                (df_shifted["id"] == selected_id) & (df_shifted["frame"] >= new_t0), "y"
            ] += delta_y

            # Get vx_mean and vy_mean
            vx_mean = (
                df_shifted[
                    (df_shifted["id"] == selected_id)
                    & (df_shifted["frame"] > new_max_frame - 15)
                ]["vx"]
                .mean()
                .round(2)
            )
            vy_mean = (
                df_shifted[
                    (df_shifted["id"] == selected_id)
                    & (df_shifted["frame"] > new_max_frame - 15)
                ]["vy"]
                .mean()
                .round(2)
            )
            v_angle = np.degrees(np.arctan2(vy_mean, vx_mean)).round(2)

            closest = df_shifted[df_shifted["id"] == selected_id]["closest"].values[0]

            # Add missing frames by shifting the frame
            columns = df_shifted.columns
            for i in range(1, abs(shift) + 1):
                x = (
                    df_shifted[
                        (df_shifted["id"] == selected_id)
                        & (df_shifted["frame"] == new_max_frame)
                    ]["x"].values[0]
                    + vx_mean / 15 * i
                ).round(2)
                y = (
                    df_shifted[
                        (df_shifted["id"] == selected_id)
                        & (df_shifted["frame"] == new_max_frame)
                    ]["y"].values[0]
                    + vy_mean / 15 * i
                ).round(2)

                df_add = pd.DataFrame(
                    [
                        [
                            selected_id,
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
                    ],
                    columns=columns,
                )

                df_shifted = pd.concat([df_shifted, df_add])

            df_shifted = df_shifted.sort_values(["frame", "id"]).reset_index(drop=True)

        return df_shifted

    def shift_backward(self, shift):
        """Shift selected candidate DataFrame backward in time.

        Moves the selected movement sequence later in time by extrapolating
        new frames at the beginning based on velocity patterns and removing
        frames from the end.

        Args:
            shift (int): Number of frames to shift backward (must be positive).
                Positive values move the movement later in time.

        Returns:
            pd.DataFrame or None: Modified DataFrame with shifted movement, or None
                if insufficient data exists for the requested shift.

        Raises:
            AssertionError: If shift is not positive (shift <= 0).

        Note:
            Uses historical velocity data to extrapolate realistic movement
            patterns for the pre-movement period.
        """
        assert shift > 0
        df_shifted = self.selected.copy()
        t0 = df_shifted[df_shifted["selected"]]["frame"].min()
        t1 = df_shifted[df_shifted["selected"]]["frame"].max()
        max_frame = df_shifted["frame"].max()
        new_t0 = t0 + shift

        if max_frame - t1 < shift:
            return None

        for class_type in ["offense", "defense"]:
            column = "selected" if class_type == "offense" else "selected_def"

            # Get the selected player id
            selected_id = int(df_shifted[df_shifted[column]]["id"].values[0])

            # Get vx_mean and vy_mean
            vx_mean = (
                df_shifted[
                    (df_shifted["id"] == selected_id)
                    & (df_shifted["frame"] < t0)
                    & (df_shifted["frame"] >= t0 - shift)
                ]["vx"]
                .mean()
                .round(2)
            )
            vy_mean = (
                df_shifted[
                    (df_shifted["id"] == selected_id)
                    & (df_shifted["frame"] < t0)
                    & (df_shifted["frame"] >= t0 - shift)
                ]["vy"]
                .mean()
                .round(2)
            )
            v_angle = np.degrees(np.arctan2(vy_mean, vx_mean)).round(2)

            # Get the delta_x and delta_y
            delta_x = vx_mean / 15 * shift
            delta_y = vy_mean / 15 * shift

            # Remove duplicate frames by shifting the frame
            df_shifted = df_shifted[
                ~(
                    (df_shifted["id"] == selected_id)
                    & (df_shifted["frame"] > max_frame - shift)
                )
            ]

            # Shift the frames
            df_shifted.loc[
                (df_shifted["id"] == selected_id) & (df_shifted["frame"] >= t0), "frame"
            ] += shift

            # Correct the x and y positions
            df_shifted.loc[
                (df_shifted["id"] == selected_id) & (df_shifted["frame"] >= new_t0), "x"
            ] += delta_x
            df_shifted.loc[
                (df_shifted["id"] == selected_id) & (df_shifted["frame"] >= new_t0), "y"
            ] += delta_y

            closest = df_shifted[df_shifted["id"] == selected_id]["closest"].values[0]

            # Add missing frames by shifting the frame
            columns = df_shifted.columns
            for i in range(1, shift + 1):
                x = (
                    df_shifted[
                        (df_shifted["id"] == selected_id)
                        & (df_shifted["frame"] == t0 - 1)
                    ]["x"].values[0]
                    + vx_mean / 15 * i
                ).round(2)
                y = (
                    df_shifted[
                        (df_shifted["id"] == selected_id)
                        & (df_shifted["frame"] == t0 - 1)
                    ]["y"].values[0]
                    + vy_mean / 15 * i
                ).round(2)

                df_add = pd.DataFrame(
                    [
                        [
                            selected_id,
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
                    ],
                    columns=columns,
                )

                df_shifted = pd.concat([df_shifted, df_add])

            df_shifted = df_shifted.sort_values(["frame", "id"]).reset_index(drop=True)

        return df_shifted

    def adjust_disc_positions(self):
        """Adjust disc positions in all scenarios to maintain consistency.

        Ensures that disc positions are correctly aligned with ball possession
        throughout all generated scenarios. Interpolates disc movement between
        known holder positions and handles possession transitions.

        The method performs the following operations:
            1. Identifies the current disc holder in each frame
            2. Resolves conflicts when multiple players appear to hold the disc
            3. Sets disc position to match the holder's position
            4. Interpolates disc positions for frames without clear possession

        Note:
            Modifies all scenario DataFrames in self.scenarios in-place to ensure disc
            positions follow the holder and are smoothly interpolated between
            possession changes.
        """
        for shift, scenario_df in self.scenarios.items():
            scenario_df = scenario_df.sort_values(
                by="frame", ascending=False
            ).reset_index(drop=True)

            selected_id = int(scenario_df[scenario_df["selected"]]["id"].values[0])

            temp = None
            for frame, frame_data in scenario_df.groupby("frame"):
                if len(frame_data[frame_data["holder"]]) == 1:
                    temp = frame_data[frame_data["holder"]]["id"].values[0]
                elif len(frame_data[frame_data["holder"]]) >= 2:
                    scenario_df.loc[
                        (scenario_df["frame"] == frame)
                        & (scenario_df["id"] != selected_id),
                        "holder",
                    ] = False
                    temp = frame_data[frame_data["holder"]]["id"].values[0]
                else:
                    scenario_df.loc[
                        (scenario_df["frame"] == frame) & (scenario_df["id"] == temp),
                        "holder",
                    ] = True

            scenario_df = scenario_df.sort_values(by=["frame", "id"]).reset_index(
                drop=True
            )

            # Initialize the disc position
            scenario_df.loc[
                (scenario_df["class"] == "disc")
                & (scenario_df["frame"] != scenario_df["frame"].max()),
                "x",
            ] = None
            scenario_df.loc[
                (scenario_df["class"] == "disc")
                & (scenario_df["frame"] != scenario_df["frame"].max()),
                "y",
            ] = None

            # Get the frame where the holder has the disc
            holder_frame = scenario_df[scenario_df["holder"]]["frame"].unique()

            # Set the disc position to the holder's position
            for frame in holder_frame:
                scenario_df.loc[
                    (scenario_df["class"] == "disc") & (scenario_df["frame"] == frame),
                    "x",
                ] = scenario_df.loc[
                    (scenario_df["holder"]) & (scenario_df["frame"] == frame),
                    "x",
                ].values[
                    0
                ]
                scenario_df.loc[
                    (scenario_df["class"] == "disc") & (scenario_df["frame"] == frame),
                    "y",
                ] = scenario_df.loc[
                    (scenario_df["holder"]) & (scenario_df["frame"] == frame),
                    "y",
                ].values[
                    0
                ]

            # Interpolate the missing disc position
            scenario_df.loc[scenario_df["class"] == "disc", ["x", "y"]] = (
                scenario_df.loc[scenario_df["class"] == "disc", ["x", "y"]]
                .interpolate(method="linear", limit_direction="both")
                .round(2)
            )

            self.scenarios[shift] = scenario_df

    # def calc_wuppcf(self, scenario_df):
    #     pass

    def get_v_frame_series(
        self,
        scenario_df,
        wuppcf,
        xgrid=np.arange(1, 94, 2),
        ygrid=np.arange(37 / 19 / 2, 37, 37 / 19),
    ):
        """Calculate V_frame series for all frames in the scenario.

        Args:
            scenario_df (pd.DataFrame): Scenario data with player positions and movements.
            wuppcf (np.ndarray): Pre-computed wUPPCF values for evaluation.
            xgrid (np.ndarray, optional): X-coordinate grid for field discretization.
                Defaults to np.arange(1, 94, 2).
            ygrid (np.ndarray, optional): Y-coordinate grid for field discretization.
                Defaults to np.arange(37/19/2, 37, 37/19).

        Returns:
            list: List of V_frame values for each frame in the scenario.
        """
        unique_frames = sorted(scenario_df["frame"].unique())
        v_frame_list = []
        for frame in unique_frames:
            frame_df = scenario_df[scenario_df["frame"] == frame]
            off_df = (
                frame_df[(frame_df["class"] == "offense") & (~frame_df["holder"])]
                .sort_values(by="id")
                .reset_index(drop=True)
            )
            disc_pos = frame_df[frame_df["class"] == "disc"][["x", "y"]].values[0]
            off_ids = off_df["id"].values
            selected_id = frame_df[frame_df["selected"]]["id"].values
            if selected_id.size == 0:
                continue
            selected_index = np.where(off_ids == selected_id[0])[0][0]
            v_frame = self.calc_v_frame(
                wuppcf[selected_index, frame - min(unique_frames) - 1, :, :],
                off_df.iloc[selected_index][["x", "y", "vx", "vy"]],
                xgrid,
                ygrid,
                disc_pos,
                disc_speed=10,
            )
            v_frame_list.append(v_frame)

        return v_frame_list

    def calc_v_frame(self, wuppcf, off_row, xgrid, ygrid, disc_pos, disc_speed=10):
        """Calculate V_frame value for a single offensive player.

        Computes the value of a frame based on the weighted ultimate probabilistic
        player control field (wUPPCF) and player positioning relative to the disc.

        Args:
            wuppcf (np.ndarray): 2D array of pre-computed wUPPCF values.
            off_row (pd.Series): Series containing player data with keys:
                - x, y: Current position coordinates
                - vx, vy: Current velocity components
            xgrid (np.ndarray): X-coordinate grid for field discretization.
            ygrid (np.ndarray): Y-coordinate grid for field discretization.
            disc_pos (tuple): Current disc position as (x, y) coordinates.
            disc_speed (float, optional): Speed of disc movement in m/s. Defaults to 10.

        Returns:
            float: V_frame value representing the strategic value of the position.
        """
        x_p, y_p, vx, vy = off_row
        x_d, y_d = disc_pos

        A = vx**2 + vy**2 - disc_speed**2
        B = 2 * (vx * (x_p - x_d) + vy * (y_p - y_d))
        C = (x_p - x_d) ** 2 + (y_p - y_d) ** 2

        D = B**2 - 4 * A * C

        t1 = (-B + np.sqrt(D)) / (2 * A)
        t2 = (-B - np.sqrt(D)) / (2 * A)
        t = max(t1, t2)

        x_t = x_p + vx * t
        y_t = y_p + vy * t

        radius = np.sqrt((vx * t * 0.5) ** 2 + (vy * t * 0.5) ** 2)

        # Create a 2D grid of coordinates
        X, Y = np.meshgrid(xgrid, ygrid)

        # Create a mask for points within the radius
        mask = (X - x_t) ** 2 + (Y - y_t) ** 2 <= radius**2

        # Check if the mask results in any valid points
        if not np.any(mask):
            return 0.0

        v_frame = np.mean(np.flipud(wuppcf)[mask]).round(3)

        return v_frame

    def calc_v_scenario(self, v_frame_list):
        """Calculate V_scenario value from a list of V_frame values.

        Computes the scenario-level value by applying a moving average filter
        and taking the maximum value, representing the peak strategic opportunity.

        Args:
            v_frame_list (list): List of V_frame values for each frame in the scenario.

        Returns:
            float: V_scenario value representing the maximum strategic value
                achieved during the scenario.
        """
        v_scenario = (
            np.convolve(v_frame_list, np.ones(15) / 15, mode="same").max().round(3)
        )
        return v_scenario


def main():
    """Main function to demonstrate VTCS analysis workflow.

    Loads ultimate frisbee tracking data, detects movement candidates,
    allows user selection of a candidate, generates counterfactual scenarios,
    and creates visualizations for analysis.

    The workflow includes:
        1. Load tracking data from CSV
        2. Detect movement candidates
        3. Interactive candidate selection
        4. Generate temporal shift scenarios
        5. Create visualization videos
        6. Evaluate scenario performance

    Example:
        Run the script directly to start the interactive analysis:
            $ python VTCS.py
    """
    # Example usage
    file_name = "data/input/UltimateTrack/1_1_2.csv"
    ultimate_track_df = pd.read_csv(file_name)
    vtcs = VTCS(ultimate_track_df)

    vtcs.detect_candidates()
    # for candidate_id, candidate_df in vtcs.candidates.items():
    #     vis.plot_play(candidate_df, save_path=f"output/{candidate_id}")

    if vtcs.candidates:
        # select a candidate
        while True:
            print("Available candidates:" f" {', '.join(vtcs.candidates.keys())}")
            candidate_index = input("Enter candidate index to select (e.g., '1-1'): ")
            if candidate_index == "q":
                print("Exiting.")
                sys.exit(0)
            if candidate_index in vtcs.candidates:
                vtcs.selected = vtcs.candidates[candidate_index]
                print(
                    "Selected candidate:",
                    vtcs.selected.attrs["movement_player_id"],
                    "from frame",
                    vtcs.selected.attrs["movement_start_frame"],
                    "to",
                    vtcs.selected.attrs["movement_end_frame"],
                )
                break
            else:
                print("Invalid candidate index. Please try again.")

        vtcs.generate_scenarios()
        # for shift, scenario_df in tqdm(vtcs.scenarios.items()):
        #     vis.plot_play(scenario_df, save_path=f"output/scenario/{shift}.mp4")

        evaluation_results = vtcs.evaluate(
            file_name.split("/")[-1].split(".")[0], candidate_index
        )

        print("Evaluation Results:")
        print("V_frame:", evaluation_results["v_frame"])
        print("V_scenario:", evaluation_results["v_scenario"])
        print("V_timing:", evaluation_results["v_timing"])
        print("Best timing shift:", evaluation_results["best_timing"])


if __name__ == "__main__":
    main()
