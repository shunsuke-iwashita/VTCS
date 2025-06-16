import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from core.frame import Frame
from core.play import Play
from core.player import Player
from core.set import Set


def detect_initiation(player_df, accel_thres=4.0, angle_thres=90, prev_holder_window=30):
    prev_holder = player_df['holder'].rolling(window=prev_holder_window, min_periods=1).max().shift(1, fill_value=False)
    prev_holder = prev_holder.astype(bool)
    cond = (
        (player_df['class'] == 'offense') &
        (player_df['a_mag'] > accel_thres) &
        (~player_df['holder']) &
        (~prev_holder) &
        (player_df['diff_v_a_angle'] < angle_thres)
    )
    return player_df[cond].frame.values

def expansion_movement(df, candidate_frames, min_cont_frames=2, angle_window=5):
    """
    各候補フレームごとに
    - diff_v_angle <= 20
    - v_mag > 3
    - holder == False
    - v_angleが過去angle_windowフレームの中央値±90°以内
    をすべて満たす限り連続拡張し、movement区間を形成
    """
    intervals = []
    n = len(candidate_frames)
    if n == 0:
        return intervals
    start = candidate_frames[0]
    end = start
    for i in range(1, n):
        prev_idx = candidate_frames[i-1]
        curr_idx = candidate_frames[i]
        # 連続フレームでなければリセット
        if curr_idx != prev_idx + 1:
            if end - start + 1 >= min_cont_frames:
                intervals.append((start, end))
            start = curr_idx
            end = curr_idx
            continue
        # 次フレームのデータで判定
        next_frame_data = df[df['frame'] == curr_idx]
        if next_frame_data.empty:
            continue
        # 過去windowフレームのv_angleの中央値
        window_frames = [f for f in candidate_frames if start <= f <= curr_idx]
        past_angles = df[df['frame'].isin(window_frames)]['v_angle']
        if len(past_angles) == 0:
            continue
        median_angle = np.median(past_angles)
        angle_diff = np.abs(((next_frame_data['v_angle'].values[0] - median_angle + 180) % 360) - 180)
        # 全条件を満たせば拡張
        if (next_frame_data['diff_v_angle'].values[0] <= 20 and
            next_frame_data['v_mag'].values[0] > 3 and
            not next_frame_data['holder'].values[0] and
            angle_diff <= 90):
            end = curr_idx
        else:
            if end - start + 1 >= min_cont_frames:
                intervals.append((start, end))
            start = curr_idx
            end = curr_idx
    if end - start + 1 >= min_cont_frames:
        intervals.append((start, end))

    if 'selected' in df.columns:
        for id_val in df['id'].unique():
            obj_data = df[df['id'] == id_val].copy()
            selected_frames = obj_data[obj_data['selected'] == True]['frame'].values
            first_frames = [frame for frame in selected_frames if frame-1 not in selected_frames]
            for frame in first_frames:
                i = frame
                while i > 0:
                    cond_prev = obj_data[(obj_data['frame'] == i-1)]
                    cond_curr = obj_data[(obj_data['frame'] == i)]
                    if cond_prev.empty or cond_curr.empty:
                        break
                    if cond_prev['selected'].values[0] == True:
                        break
                    elif (
                        cond_prev['v_mag'].values[0] - cond_curr['v_mag'].values[0] < 0.05 and
                        cond_prev['v_mag'].values[0] > 0.05 and
                        cond_prev['selected'].values[0] == False
                    ):
                        # selectedを前方に拡張
                        df.loc[(df['id'] == id_val) & (df['frame'] == i-1) & (df['frame'] != 0), 'selected'] = True
                    else:
                        break
                    i -= 1
    print(intervals)
    return intervals


def exclusion_movement(intervals, min_length=5, min_interval=30):
    filtered = []
    last_end = -min_interval
    for (start, end) in intervals:
        if end - start + 1 < min_length:
            continue
        if start - last_end <= min_interval and filtered:
            filtered[-1] = (filtered[-1][0], end)
        else:
            filtered.append((start, end))
        last_end = end
    return filtered

def compute_features(df):
    # 速度・加速度の大きさ
    if 'v_mag' not in df.columns:
        df['v_mag'] = np.sqrt(df['vx']**2 + df['vy']**2)
    if 'a_mag' not in df.columns:
        df['a_mag'] = np.sqrt(df['ax']**2 + df['ay']**2)
    # 速度・加速度の方向
    if 'v_angle' not in df.columns:
        df['v_angle'] = np.degrees(np.arctan2(df['vy'], df['vx']))
    if 'a_angle' not in df.columns:
        df['a_angle'] = np.degrees(np.arctan2(df['ay'], df['ax']))
    # 速度と加速度の方向差
    if 'diff_v_a_angle' not in df.columns:
        diff = df['v_angle'] - df['a_angle']
        # -180~180に正規化
        diff = (diff + 180) % 360 - 180
        df['diff_v_a_angle'] = np.abs(diff)
    # 1フレーム前との速度の方向差
    if 'diff_v_angle' not in df.columns:
        diff = df['v_angle'].diff().abs().fillna(0)
        df['diff_v_angle'] = np.abs(diff)
    return df

def detect_movement_for_player(player: Player, class_name='offense', **kwargs):
    # offenseのみ検出
    if hasattr(player, 'class_name') and player.class_name != class_name:
        return []
    df = pd.DataFrame([
        {'frame': f, **state} for f, state in player.states.items()
    ]).sort_values('frame').reset_index(drop=True)
    df = compute_features(df)
    candidate_frames = detect_initiation(df, **kwargs)
    intervals = expansion_movement(df, candidate_frames)
    final_intervals = exclusion_movement(intervals)
    return [{'id': player.id, 'start_frame': start, 'end_frame': end} for (start, end) in final_intervals]


def detect_all_movements(set_obj: Set, **kwargs):
    plays = []
    for pid, player in set_obj.players.items():
        play_intervals = detect_movement_for_player(player, **kwargs)
        for interval in play_intervals:
            play_frames = [f for f in set_obj.frames if interval['start_frame'] <= f.idx <= interval['end_frame']]
            plays.append(
                Play(
                    initiator_id=pid,
                    start_frame=interval['start_frame'],
                    end_frame=interval['end_frame'],
                    frames=play_frames
                )
            )
    return plays

def build_set_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    players = {pid: Player(pid) for pid in df['id'].unique()}
    frames = []
    for frame_num in sorted(df['frame'].unique()):
        frame_df = df[df['frame'] == frame_num]
        frame = Frame(frame_num)
        for _, row in frame_df.iterrows():
            pid = int(row['id'])
            state_dict = {col: row[col] for col in df.columns if col not in ['frame', 'id']}
            players[pid].add_state(frame_num, state_dict)
            frame.add_player_state(pid, state_dict)
        frames.append(frame)
    set_obj = Set(frames, players)
    return set_obj

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="UltimateTrack CSVファイルへのパス")
    args = parser.parse_args()

    # 1. 毎回CSVからSetを構築
    input_path = f'data/input/UltimateTrack/{args.input}.csv'
    set_obj = build_set_from_csv(input_path)
    print(f"Setオブジェクトを生成しました: {args.input}")

    # 2. movement検出
    plays = detect_all_movements(set_obj)
    print(f"検出されたplay数: {len(plays)}")

    # 必要ならプレイ区間情報を保存
    play_info = [{'initiator_id': p.initiator_id, 'start_frame': p.start_frame, 'end_frame': p.end_frame} for p in plays]
    output_csv = os.path.splitext(input_path)[0] + "_play_segments.csv"
    pd.DataFrame(play_info).to_csv(output_csv, index=False)
    print(f"play区間情報を {output_csv} に保存しました。")
