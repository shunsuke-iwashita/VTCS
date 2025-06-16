class Play:
    def __init__(
        self,
        play_id: str,
        frames: list,                  # list[Frame]：プレー全体の時系列Frame
        metadata: dict = None          # 例：開始フレーム、攻守判定、パス先id等
    ):
        self.play_id = play_id
        self.frames = frames
        self.metadata = metadata or {}

        self.scenarios = []            # 反実仮想Scenario（shiftごと）を格納
        self.vtcs = None               # VTCSスコア（必要に応じて）
        # 必要に応じてプレー全体の他特徴量も

    def add_scenario(self, scenario):
        self.scenarios.append(scenario)

    def get_frame(self, idx):
        """フレーム番号からFrameを取得"""
        for f in self.frames:
            if f.frame_idx == idx:
                return f
        return None

    def all_player_ids(self):
        """このプレーに登場する全選手id"""
        return set(pid for frame in self.frames for pid in frame.players.keys())

    def calc_play_features(self):
        """プレー全体の特徴量・集計を計算（任意）"""
        pass
