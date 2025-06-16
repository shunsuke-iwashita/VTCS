class Set:
    def __init__(self, plays=None, players=None, frames=None):
        self.plays = plays or []
        self.players = players or {}   # id: Player（Set全体でのPlayer）
        self.frames = frames or []     # Set全体のFrame

    def add_play(self, play):
        self.plays.append(play)

    def add_player(self, player):
        self.players[player.id] = player

    def add_frame(self, frame):
        self.frames.append(frame)