import itertools
import sys

import numpy as np
import pandas as pd

import VTCS_visualize as vis


class VTCS:
    def __init__(self, ultimate_track_df):
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
        """
        # Detect movement initiation from the play DataFrame.
        self.detect_movement_initiation()  # 例: play DataFrameを更新する関数
        self.deselect_movement_initiation()  # 検出された動き出しを除外する関数
        self.extract_movement_candidates()  # 例: [df1, df2, ...]

    def generate_scenarios(self, shifts=range(-15, 16)):
        """
        選択した候補DataFrame(self.selected)から各シフトの反実仮想シナリオを生成し、dictで格納。
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

    def evaluate(self):
        """
        各シナリオDataFrameごとにV_frame, V_scenarioを計算し、V_timingも算出。
        """
        self.v_frame = {}
        self.v_scenario = {}

        for shift, scenario_df in self.scenarios.items():
            v_frame_list = calc_v_frame_series(scenario_df)  # 例: [float, ...]を返す
            v_scenario_val = calc_v_scenario_from_v_frame(
                v_frame_list
            )  # 例: floatを返す

            self.v_frame[shift] = v_frame_list
            self.v_scenario[shift] = v_scenario_val

        # V_timing計算
        v_actual = self.v_scenario.get(0, None)
        v_best = max([v for k, v in self.v_scenario.items() if k != 0])
        self.v_timing = (
            v_actual - v_best if v_actual is not None and v_best is not None else None
        )

        return {
            "v_frame": self.v_frame,  # {shift: [v1, v2, ...]}
            "v_scenario": self.v_scenario,  # {shift: value}
            "v_timing": self.v_timing,  # float
        }

    def detect_movement_initiation(self, v_threshold: float=3.0, a_threshold: float=4.0):
        """
        動き出しの検出を行い、play DataFrameを更新。
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
        """Deselect candidates based on proximity and running direction."""

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
        """
        全選手分のデータを、selected==Trueの連続区間ごとに
        「その区間＋前後windowフレーム」を含めて抽出し、リストで返す。
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

    # def fill_holder(self):
    #     """
    #     選択した候補DataFrameのholder列をTrueに設定する。
    #     """
    #     # frameごとのholder=Trueを管理
    #     holder_id = None
    #     for frame, group in self.selected.groupby("frame"):
    #         holders = group[group["holder"]]
    #         if len(holders) == 1:
    #             # ちょうど1つ → そのidを記憶
    #             holder_id = holders["id"].iloc[0]
    #         elif len(holders) == 0:
    #             # 0なら直近のholder_idを使う
    #             if holder_id is not None:
    #                 mask = (self.selected["frame"] == frame) & (
    #                     self.selected["id"] == holder_id
    #                 )
    #                 self.selected.loc[mask, "holder"] = True
    #                 mask_ = (self.selected["frame"] == frame) & (
    #                     self.selected["class"] == "disc"
    #                 )
    #                 self.selected.loc[mask_, ["x", "y"]] = self.selected.loc[
    #                     mask, ["x", "y"]
    #                 ].values

    def shift_forward(self, shift):
        """
        選択した候補DataFrameを前方にシフトする。
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
        """
        選択した候補DataFrameを後方にシフトする。
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
                (df_shifted["id"] == selected_id) & (df_shifted["frame"] > t0), "frame"
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
                        & (df_shifted["frame"] == t0)
                    ]["x"].values[0]
                    - vx_mean / 15 * i
                ).round(2)
                y = (
                    df_shifted[
                        (df_shifted["id"] == selected_id)
                        & (df_shifted["frame"] == t0)
                    ]["y"].values[0]
                    - vy_mean / 15 * i
                ).round(2)

                df_add = pd.DataFrame(
                    [
                        [
                            selected_id,
                            t0 + i,
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
        """
        選択した候補DataFrameのディスク位置を調整する。
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


def main():
    # Example usage
    ultimate_track_df = pd.read_csv("data/input/UltimateTrack/1_1_2.csv")
    vtcs = VTCS(ultimate_track_df)

    vtcs.detect_candidates()
    # for candidate_id, candidate_df in vtcs.candidates.items():
    #     vis.plot_play(candidate_df, save_path=f"output/{candidate_id}")

    if vtcs.candidates:
        # select a candidate
        while True:
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
        for shift, scenario_df in vtcs.scenarios.items():
            vis.plot_play(scenario_df, save_path=f"output/scenario/{shift}.mp4")

        evaluation_results = vtcs.evaluate()


if __name__ == "__main__":
    main()
    main()
