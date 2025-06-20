import numpy as np
import pandas as pd

class VTCS:
    def __init__(self, ultimate_track_df):
        self.play = ultimate_track_df
        self.candidates = []   # list: DataFrame
        self.selected = None   # DataFrame
        self.scenarios = {}    # dict: {shift: DataFrame of scenario}

        # --- 評価指標（初期化時はNoneや空で持つ） ---
        self.v_frame = None      # dict: {shift: list of values}
        self.v_scenario = None   # dict: {shift: value}
        self.v_timing = None     # float

        # 必要なら
        self.optimal_shift = None
        self.result_summary = None

        self.play['v_mag'] = np.sqrt(self.play['vx']**2 + self.play['vy']**2)
        self.play['a_mag'] = np.sqrt(self.play['ax']**2 + self.play['ay']**2)
        self.play['v_angle'] = np.arctan2(self.play['vy'], self.play['vx']) * 180 / np.pi
        self.play['a_angle'] = np.arctan2(self.play['ay'], self.play['ax']) * 180 / np.pi
        self.play['diff_v_angle'] = self.play.groupby('id')['v_angle'].diff().abs().apply(lambda x: min(x, 360 - x) if pd.notnull(x) else x)

    def detect_candidates(self):
        """
        動き出し候補をDataFrameごとに抽出してself.candidates（リスト）に格納。
        """
        # Detect movement initiation from the play DataFrame.
        self.detect_movement_initiation()  # 例: play DataFrameを更新する関数
        self.deselect_movement_initiation()  # 検出された動き出しを除外する関数
        self.candidates = detect_movement_candidates(self.play)  # 例: [df1, df2, ...]
        return self.candidates

    def select_candidate(self, candidate_idx):
        """
        UI等からインデックスで候補を指定し、そのDataFrameをself.selectedにセット。
        """
        self.selected = self.candidates[candidate_idx]
        self._generate_scenarios()

    def _generate_scenarios(self, shifts=range(-15, 16)):
        """
        選択した候補DataFrame(self.selected)から各シフトの反実仮想シナリオを生成し、dictで格納。
        """
        self.scenarios = {}
        for shift in shifts:
            scenario_df = create_scenario_df(self.selected, shift)  # 例: DataFrame返す関数
            self.scenarios[shift] = scenario_df

    def evaluate(self):
        """
        各シナリオDataFrameごとにV_frame, V_scenarioを計算し、V_timingも算出。
        """
        self.v_frame = {}
        self.v_scenario = {}

        for shift, scenario_df in self.scenarios.items():
            v_frame_list = calc_v_frame_series(scenario_df)  # 例: [float, ...]を返す
            v_scenario_val = calc_v_scenario_from_v_frame(v_frame_list)  # 例: floatを返す

            self.v_frame[shift] = v_frame_list
            self.v_scenario[shift] = v_scenario_val

        # V_timing計算
        v_actual = self.v_scenario.get(0, None)
        v_best = max([v for k, v in self.v_scenario.items() if k != 0])
        self.v_timing = v_actual - v_best if v_actual is not None and v_best is not None else None

        return {
            'v_frame': self.v_frame,         # {shift: [v1, v2, ...]}
            'v_scenario': self.v_scenario,   # {shift: value}
            'v_timing': self.v_timing        # float
        }

    def detect_movement_initiation(self, v_threshold=3.0, a_threshold=4.0):
        """
        動き出しの検出を行い、play DataFrameを更新。
        """
        # ---------- detect movement initiation ----------
        self.play['selected'] = False
        self.play['prev_holder'] = self.play.groupby('id')['holder'].shift(30)
        angle_diff = np.abs((self.play['v_angle'] - self.play['a_angle'] + 180) % 360 - 180)
        # Define the selection criteria
        selection_criteria = (
            (self.play['frame'] > 1) &
            (self.play['class'] == 'offense') &
            (self.play['holder'] == False) &
            (self.play['prev_holder'] != True) &
            (self.play['a_mag'] > a_threshold) &
            (angle_diff< 90)
        )
        self.play.loc[selection_criteria, 'selected'] = True

        # ---------- forward expansion ----------
        def circular_average(angles):
            radians = np.deg2rad(angles)
            sin_average = np.mean(np.sin(radians))
            cos_average = np.mean(np.cos(radians))
            average_radian = np.arctan2(sin_average, cos_average)
            average_angle = np.rad2deg(average_radian)
            average_angle = average_angle % 360
            return average_angle

        for id, id_df in self.play.groupby('id'):
            id_df = id_df.sort_values('frame').reset_index()
            frames = id_df['frame'].values
            v_angles = id_df['v_angle'].values
            selected = id_df['selected'].values
            diff_v_angle = id_df['diff_v_angle'].values
            v_mag = id_df['v_mag'].values
            holder = id_df['holder'].values

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
                    (diff_v_angle[ni] <= 20) and
                    (v_mag[ni] > v_threshold) and
                    (not holder[ni]) and
                    (angle_diff <= 90)
                )
                selected[ni] = cond

            self.play.loc[id_df['index'], 'selected'] = selected

        # ---------- deselect candidates based on length ----------
        def update_selected_and_length(group):
            group = group.sort_values('frame').reset_index(drop=True)
            selected = group['selected'].values
            frames = group['frame'].values

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

            group['length'] = length
            group.loc[(group['length'] < 15) | (group['length'] > 75), 'selected'] = False
            return group

        self.play = self.play.groupby('id', group_keys=False).apply(update_selected_and_length)

        # ---------- backward expansion ----------
        self.play.set_index(['id', 'frame'], inplace=True)

        max_frame = self.play.index.get_level_values('frame').max()
        min_frame = self.play.index.get_level_values('frame').min()

        for frame in range(max_frame, min_frame, -1):
            selected_idx = self.play.index[self.play['selected'] & (self.play.index.get_level_values('frame') == frame)]
            for idx in selected_idx:
                id = idx[0]
                prev_idx = (id, frame - 1)
                if prev_idx in self.play.index:
                    if (
                        not self.play.at[prev_idx, 'selected'] and
                        self.play.at[prev_idx, 'v_mag'] > 0.05 and
                        self.play.at[prev_idx, 'v_mag'] - self.play.at[idx, 'v_mag'] < 0.05
                    ):
                        self.play.at[prev_idx, 'selected'] = True

        self.play.reset_index(inplace=True)

    def deselect_movement_initiation(self):
        # Based on proximity
        pass


def main():
    # Example usage
    ultimate_track_df = pd.read_csv('data/input/UltimateTrack/1_1_2.csv')
    vtcs = VTCS(ultimate_track_df)

    vtcs.detect_candidates()
    print("Candidates detected:", vtcs.candidates)

    if vtcs.candidates:
        vtcs.select_candidate(0)  # Select the first candidate
        print("Selected candidate:", vtcs.selected)

        vtcs._generate_scenarios()
        print("Generated scenarios:", vtcs.scenarios)

        evaluation_results = vtcs.evaluate()
        print("Evaluation results:", evaluation_results)

if __name__ == "__main__":
    main()