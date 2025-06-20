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
        """Detect movement initiation and update ``self.play``."""

        # initialise selection flag
        self.play['selected'] = False

        # previous possession state
        self.play['prev_holder'] = self.play.groupby('id')['holder'].shift(30)

        # base conditions for the first frame of a candidate sequence
        criteria = (
            (self.play['frame'] > 1) &
            (self.play['class'] == 'offense') &
            (~self.play['holder']) &
            (self.play['prev_holder'] != True) &
            (self.play['a_mag'] > a_threshold) &
            (abs((self.play['v_angle'] - self.play['a_angle'] + 180) % 360 - 180) < 90)
        )
        self.play.loc[criteria, 'selected'] = True

        # ---------- forward expansion per id ----------
        for pid, grp in self.play.groupby('id'):
            grp = grp.sort_values('frame')
            idx = grp.index
            selected = grp['selected'].to_numpy()

            v_angles = np.deg2rad(grp['v_angle'].to_numpy())
            diff_v = grp['diff_v_angle'].to_numpy()
            v_mag = grp['v_mag'].to_numpy()
            holder = grp['holder'].to_numpy()

            sin_sum = 0.0
            cos_sum = 0.0
            count = 0

            for i in range(len(grp) - 1):
                if selected[i]:
                    sin_sum += np.sin(v_angles[i])
                    cos_sum += np.cos(v_angles[i])
                    count += 1

                    j = i + 1
                    mean_angle = np.arctan2(sin_sum / count, cos_sum / count)
                    ang_diff = abs(np.rad2deg(np.arctan2(np.sin(v_angles[j] - mean_angle),
                                                        np.cos(v_angles[j] - mean_angle))))

                    if (diff_v[j] <= 20 and v_mag[j] > v_threshold and not holder[j] and ang_diff <= 90):
                        selected[j] = True
                    else:
                        sin_sum = 0.0
                        cos_sum = 0.0
                        count = 0
                else:
                    sin_sum = 0.0
                    cos_sum = 0.0
                    count = 0

            self.play.loc[idx, 'selected'] = pd.Series(selected, index=idx)

        # ---------- remove too short/long sequences ----------
        def _set_seq_length(df: pd.DataFrame) -> pd.DataFrame:
            df = df.sort_values('frame').reset_index(drop=True)
            sel = df['selected'].to_numpy()
            frames = df['frame'].to_numpy()
            length = np.zeros_like(sel, dtype=int)

            i = 0
            while i < len(sel):
                if sel[i]:
                    j = i
                    while j + 1 < len(sel) and sel[j + 1] and frames[j + 1] == frames[j] + 1:
                        j += 1
                    seq_len = j - i + 1
                    length[i:j + 1] = seq_len
                    i = j + 1
                else:
                    i += 1

            df['length'] = length
            df.loc[(df['length'] < 15) | (df['length'] > 75), 'selected'] = False
            return df

        self.play = self.play.groupby('id', group_keys=False).apply(_set_seq_length)

        # ---------- backward expansion ----------
        self.play.sort_values(['id', 'frame'], inplace=True)
        for pid, grp in self.play.groupby('id'):
            idx = grp.index
            sel = grp['selected'].to_numpy()
            v_mag = grp['v_mag'].to_numpy()

            for i in range(len(grp) - 1, 0, -1):
                if sel[i] and not sel[i - 1]:
                    if v_mag[i - 1] > 0.05 and (v_mag[i - 1] - v_mag[i] < 0.05):
                        sel[i - 1] = True

            self.play.loc[idx, 'selected'] = pd.Series(sel, index=idx)

        pd.set_option('display.max_rows', None)
        print(self.play[self.play['id'] == 3])

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