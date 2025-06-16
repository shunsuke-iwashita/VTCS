class Frame:
    def __init__(
        self,
        frame_idx: int,
        players: list,            # list[Player]またはdict[player_id, Player]
        meta: dict = None         # 追加メタ情報（任意）
    ):
        self.frame_idx = frame_idx          # フレーム番号
        self.players = {p.id: p for p in players}  # id: Player辞書で管理（高速アクセス）
        self.meta = meta or {}             # 例: 経過時間やイベント情報等

        self.v_frame = {}   # {player_id: v_frame値, ...}
        self.rank = {}      # {player_id: ランク, ...}（必要に応じて）

    def set_v_frame(self, player_id, value, rank=None):
        self.v_frame[player_id] = value
        if rank is not None:
            self.rank[player_id] = rank

    def get_player(self, player_id):
        return self.players.get(player_id, None)

    def all_players(self):
        return list(self.players.values())