class Play:
    def __init__(self, initiator_id, start_frame, end_frame, frames=None, players=None, scenarios=None):
        self.initiator_id = initiator_id
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.frames = frames or []     # Play内のFrame列
        self.players = players or {}   # Play内で出場したPlayer
        self.scenarios = scenarios or []

    def add_scenario(self, scenario):
        self.scenarios.append(scenario)

    def add_player(self, player):
        self.players[player.id] = player

    def add_frame(self, frame):
        self.frames.append(frame)