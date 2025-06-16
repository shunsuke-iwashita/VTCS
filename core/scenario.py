class Scenario:
    def __init__(self, players):
        self.players = players  # id: Player

    def get_player(self, pid):
        return self.players.get(pid)