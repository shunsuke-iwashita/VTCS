# /core/pipeline.py

from .play import Play, Scenario
from .frame import Frame
from .player import Player
# 必要に応じてutilsもimport

class VTCSPipeline:
    def __init__(self, mot_file, output_dir, config=None):
        self.mot_file = mot_file
        self.output_dir = output_dir
        self.config = config
        self.plays = []         # Playインスタンス群
        # 必要に応じて各種設定やバッファ

    def load_and_preprocess(self):
        # MOT→Player/Frame/Play構築、特徴量計算など
        pass

    def detect_initiation_and_split_play(self):
        # 動き出し検出＋プレイ区間分割
        pass

    def generate_scenarios(self):
        # Play→Scenario群（shiftごと）を生成
        pass

    def evaluate_all_scenarios(self):
        # 各Scenarioに空間評価値など計算
        pass

    def calculate_vtcs(self):
        # 各PlayのVTCS値を計算
        pass

    def export_results(self):
        # 結果CSVや中間データ出力
        pass

    def visualize_results(self):
        # 可視化・動画/ヒートマップ生成
        pass
