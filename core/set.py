from core.player import Player
from core.frame import Frame
from core.play import Play

class Set:
    def __init__(self, frames, players):
        self.frames = frames      # [Frame, ...]
        self.players = players    # id: Player

    def detect_all_movements(self, **kwargs):
        """
        全選手分のmovement検出→Play生成リストを返す
        """
        plays = []
        for pid, player in self.players.items():
            # player.detect_movementでその選手のplay候補取得
            results = player.detect_movement(**kwargs)
            for res in results:
                # 区間のFrame列を抽出
                play_frames = [f for f in self.frames if res['start_frame'] <= f.idx <= res['end_frame']]
                plays.append(Play(
                    initiator_id=pid,
                    start_frame=res['start_frame'],
                    end_frame=res['end_frame'],
                    frames=play_frames
                ))
        return plays
