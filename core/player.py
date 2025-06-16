class Player:
    def __init__(
        self, 
        id: int,
        role: str,               # 'offense', 'defense', 'disc'
        coords: tuple[float, float],  # (x, y)
        velocity: tuple[float, float],# (vx, vy)
        accel: tuple[float, float],   # (ax, ay)
        selected: bool = False,       # 動き出し検出フラグ
        holder: bool = False,         # ディスク保持フラグ
        closest_defense: int = None,  # 1対1のマーク相手id
        **kwargs                    # 他特徴量も柔軟に追加可能
    ):
        self.id = id
        self.role = role
        self.x, self.y = coords
        self.vx, self.vy = velocity
        self.ax, self.ay = accel
        self.selected = selected
        self.holder = holder
        self.closest_defense = closest_defense

        # その他の特徴量（スカラー値も随時追加可）
        for key, value in kwargs.items():
            setattr(self, key, value)