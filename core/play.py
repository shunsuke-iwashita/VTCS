class Play:
    def __init__(self, initiator_id, start_frame, end_frame, frames):
        self.initiator_id = initiator_id
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.frames = frames  # [Frame, ...]
        # 必要に応じてScenarioや評価値もここに

    # シナリオ生成、評価値計算等のメソッドを追加可能
