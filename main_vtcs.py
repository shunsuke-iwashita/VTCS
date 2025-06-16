# main_vtcs.py

from core.pipeline import VTCSPipeline

def main(input_mot_file, output_dir, config=None):
    # パイプラインの初期化
    pipeline = VTCSPipeline(input_mot_file, output_dir, config)

    # 1. データロード・前処理
    pipeline.load_and_preprocess()

    # 2. 動き出し検出＆プレイ区間分割
    pipeline.detect_initiation_and_split_play()

    # 3. シナリオ生成
    pipeline.generate_scenarios()

    # 4. 全シナリオ空間評価
    pipeline.evaluate_all_scenarios()

    # 5. VTCS算出
    pipeline.calculate_vtcs()

    # 6. 結果出力
    pipeline.export_results()
    pipeline.visualize_results()

    print(f"✔️ All processes completed. Results saved in {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='MOT形式の入力ファイル')
    parser.add_argument('--output', type=str, required=True, help='出力ディレクトリ')
    parser.add_argument('--config', type=str, default=None, help='追加設定ファイル（任意）')
    args = parser.parse_args()

    main(args.input, args.output, args.config)
