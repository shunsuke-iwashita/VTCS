class Player:
    def __init__(self, id, frames=None):
        self.id = id
        self.frames = frames or []  # [Frame, ...]

    def add_frame(self, frame):
        self.frames.append(frame)