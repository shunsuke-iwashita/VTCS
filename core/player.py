class Player:
    def __init__(self, id):
        self.id = id
        self.states = {}  # frame番号: 特徴量dict

    def as_dataframe(self):
        import pandas as pd
        return pd.DataFrame([
            {'frame': f, **state} for f, state in self.states.items()
        ])

    def add_state(self, frame, state_dict):
        self.states[frame] = state_dict