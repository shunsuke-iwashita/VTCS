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
        # Initialize the 'selected' column to False
        self.play['selected'] = False

        # Add 'prev_holder' column by shifting the 'holder' column by 30 frames for each id
        self.play['prev_holder'] = self.play.groupby('id')['holder'].shift(30)

        # Define the selection criteria
        selection_criteria = (
            # The current frame must be at least 15
            (self.play['frame'] > 1) &
            # The object must be of class 'offense'
            (self.play['class'] == 'offense') &
            # The object should not be a holder
            (self.play['holder'] == False) &
            # The object should not have been a holder 30 frames ago
            (self.play['prev_holder'] != True) &
            # The object's acceleration magnitude must be greater than the threshold
            (self.play['a_mag'] > a_threshold) &
            # The difference of the velocity angle and acceleration angle must be less than 90
            (abs((self.play['v_angle'] - self.play['a_angle'] + 180) % 360 - 180) < 90)
        )

        # Apply the selection criteria
        self.play.loc[selection_criteria, 'selected'] = True

        # Expansion of range of movement initiation forward
        for id in self.play['id'].unique():
            id_df = self.play[self.play['id'] == id]
            # Save the series of velocity angles
            v_angles = []

            for frame in range(1, id_df['frame'].max() + 1):
                # Skip if the current frame is not selected
                if not id_df.loc[id_df['frame'] == frame, 'selected'].values[0]:
                    # Reset the velocity angle list
                    v_angles = []
                    continue
                v_angles.append(id_df.loc[id_df['frame'] == frame, 'v_angle'].values[0])
                next_frame = frame + 1
                next_frame_df = id_df[id_df['frame'] == next_frame]

                # Check if the next frame exists
                if next_frame_df.empty:
                    continue

                def circular_average(angles):
                    # Convert angles to radians
                    radians = np.deg2rad(angles)
                    # Calculate the average of the sine and cosine values
                    sin_average = np.mean(np.sin(radians))
                    cos_average = np.mean(np.cos(radians))
                    # Compute the average angle in radians
                    average_radian = np.arctan2(sin_average, cos_average)
                    # Convert back to degrees
                    average_angle = np.rad2deg(average_radian)
                    # Normalize the angle to be within [0, 360)
                    average_angle = average_angle % 360
                    return average_angle

                # Calculate the angle difference
                angle_diff = abs((next_frame_df['v_angle'].values[0] - circular_average(v_angles)))
                # Ensure the angle difference is within the range [0, 180]
                angle_diff = min(angle_diff, 360 - angle_diff)

                # Check if the next frame meets the criteria
                if (
                    # The difference in velocity angle must be <= 20
                    next_frame_df['diff_v_angle'].values[0] <= 20 and
                    # The velocity magnitude must be > 3
                    next_frame_df['v_mag'].values[0] > v_threshold and
                    # The object should not be a holder
                    next_frame_df['holder'].values[0] == False and
                    # The object's velocity angle must be within the circular median of the previous angles
                    angle_diff <= 90
                ):
                    # Mark the next frame as selected
                    self.play.loc[(self.play['id'] == id) & (self.play['frame'] == next_frame), 'selected'] = True
                    id_df.loc[id_df['frame'] == next_frame, 'selected'] = True
                else:
                    self.play.loc[(self.play['id'] == id) & (self.play['frame'] == next_frame), 'selected'] = False
                    id_df.loc[id_df['frame'] == next_frame, 'selected'] = False

        def update_selected_and_length(group):
            group = group.sort_values('frame').reset_index(drop=True)
            group['length'] = 0
            start = 0
            while start < len(group):
                if group.loc[start, 'selected']:
                    end = start
                    while end + 1 < len(group) and group.loc[end + 1, 'selected'] and group.loc[end + 1, 'frame'] == group.loc[end, 'frame'] + 1:
                        end += 1
                    seq_length = end - start + 1
                    group.loc[start:end, 'length'] = seq_length
                    start = end + 1
                else:
                    start += 1
            group.loc[(group['length'] < 15) | (group['length'] > 75), 'selected'] = False
            return group

        # Deselect the movement initiation candidates based on length
        self.play = self.play.groupby('id', group_keys=False).apply(update_selected_and_length)

        # Expansion of range of movement initiation backward
        for frame in range(max(self.play['frame']), 0, -1):
            selected_rows = self.play[(self.play['frame'] == frame) & (self.play['selected'])]
            if selected_rows.empty:
                continue
            for id in selected_rows['id'].unique():
                if (
                    self.play.loc[(self.play['id'] == id) & (self.play['frame'] == frame - 1), 'selected'].values[0] == False and
                    self.play.loc[(self.play['id'] == id) & (self.play['frame'] == frame - 1), 'v_mag'].values[0] > 0.05 and
                    self.play.loc[(self.play['id'] == id) & (self.play['frame'] == frame - 1), 'v_mag'].values[0] - self.play.loc[(self.play['id'] == id) & (self.play['frame'] == frame), 'v_mag'].values[0] < 0.05
                ):
                    self.play.loc[(self.play['id'] == id) & (self.play['frame'] == frame - 1), 'selected'] = True
        pd.set_option('display.max_rows', None)  # Show all rows in the DataFrame
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