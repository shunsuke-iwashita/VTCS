import cv2
import pandas as pd
import subprocess
import os

def id_to_color(obj_id):
    """
    id=1〜14用の分かりやすい色割り当て（BGR形式）
    """
    palette = [
        (255, 0, 0),    # 青
        (0, 255, 0),    # 緑
        (0, 128, 255),  # オレンジ
        (128, 0, 128),  # 紫
        (255, 255, 0),  # 水色
        (255, 0, 255),  # マゼンタ
        (0, 255, 255),  # 黄
        (128, 128, 0),  # オリーブ
        (0, 0, 128),    # 濃紺
        (0, 128, 128),  # ティール
        (128, 0, 0),    # 茶色
        (0, 64, 128),   # くすみ青
        (64, 0, 128),   # くすみ紫
        (0, 128, 64),   # 緑系
    ]
    idx = (int(obj_id) - 1) % len(palette)
    return palette[idx]

def draw_bboxes_on_video(input_name):
    df = pd.read_csv(f'data/input/mot/{input_name}.txt', header=None)
    cap = cv2.VideoCapture(f'data/input/video/{input_name}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = f'output/{input_name}_bboxes.mp4'
    out = cv2.VideoWriter(output_path, fourcc, 15, (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bboxes = df[df[0] == frame_idx]
        for _, row in bboxes.iterrows():
            _, obj_id, x, y, w, h = row[:6]
            color = id_to_color(obj_id)
            cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
            cv2.putText(frame, str(int(obj_id)), (int(x), int(y-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        out.write(frame)
        frame_idx += 1
    cap.release()
    out.release()
    print(f"動画を書き出しました: {output_path}")
    convert_mp4_to_standard(output_path)

def convert_mp4_to_standard(input_path):
    temp_output = input_path.replace('.mp4', '_ffmpeg.mp4')
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-vcodec', 'libx264', '-acodec', 'aac', '-movflags', 'faststart',
        temp_output
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(input_path)
        os.rename(temp_output, input_path)
        print(f"✔️ ffmpeg再エンコード完了（VSCode等でも再生可能）: {input_path}")
    except Exception as e:
        print("ffmpegによる変換に失敗しました。mp4の再生に問題がある場合は手動でffmpegコマンドを実行してください。")
        print(e)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Draw bounding boxes on video from MOT data.')
    parser.add_argument('--input', type=str, required=True, help='Name of the input video file')
    args = parser.parse_args()

    draw_bboxes_on_video(args.input)
