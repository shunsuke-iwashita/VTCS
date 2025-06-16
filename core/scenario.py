class Scenario(Play):
    def __init__(
        self,
        play_id: str,
        shift: int,                       # 動き出しタイミングを何フレームずらしたか
        frames: list,                     # list[Frame]：シフト後の全フレーム
        metadata: dict = None
    ):
        super().__init__(play_id, frames, metadata)
        self.shift = shift                # シナリオ識別のためのシフト量（例: 0=実際, -5=5フレーム早い）
        self.v_scenario = None            # このScenarioの代表評価値
        self.v_frame_series = []          # 各フレームのv_frame（可視化や平滑化計算用）
        # 必要に応じて他の評価値・特徴量もここで

    def calc_v_scenario(self, method='max15avg'):
        """各フレームのv_frame系列から代表値を算出（例: 15フレーム平滑化＋最大値）"""
        v_frames = [f.v_frame.get(self.get_selected_id(f), 0.0) for f in self.frames]
        self.v_frame_series = v_frames
        # 例: 15フレーム平滑化
        import numpy as np
        smoothed = np.convolve(v_frames, np.ones(15), 'same') / 15
        self.v_scenario = float(np.max(smoothed))

    def get_selected_id(self, frame):
        """選択されたオフェンスidを取得（frameから）"""
        for pid, player in frame.players.items():
            if getattr(player, 'selected', False) and not getattr(player, 'holder', False):
                return pid
        return None
