class Frame:
    def __init__(self, idx):
        self.idx = idx
        self.players = {}  # id: PlayerState（frameごとの状態dict）
    def add_player_state(self, player_id, state_dict):
        self.players[player_id] = state_dict