import numpy as np
import random

# face indices
U, D, L, R, F, B = 0, 1, 2, 3, 4, 5

class Cube:
    def __init__(self):
        # each face is 2x2, value = color index 0-5
        self.state = np.array([
            [[i]*2 for _ in range(2)] for i in range(6)
        ], dtype=np.int8)

    def copy(self):
        c = Cube()
        c.state = self.state.copy()
        return c

    def is_solved(self):
        for i in range(6):
            if not np.all(self.state[i] == self.state[i][0][0]):
                return False
        return True

    def get_state_vector(self):
        # one-hot encode: 24 stickers x 6 colors = 144
        flat = self.state.flatten().astype(np.int32)
        onehot = np.zeros(144, dtype=np.float32)
        onehot[np.arange(24) * 6 + flat] = 1.0
        return onehot

    # move helpers
    def _rotate_face_cw(self, f):
        self.state[f] = np.rot90(self.state[f], k=-1)

    def _rotate_face_ccw(self, f):
        self.state[f] = np.rot90(self.state[f], k=1)

    # U move
    def move_U(self):
        self._rotate_face_cw(U)
        tmp = self.state[F][0].copy()
        self.state[F][0] = self.state[R][0]
        self.state[R][0] = self.state[B][0]
        self.state[B][0] = self.state[L][0]
        self.state[L][0] = tmp

    def move_Ui(self):
        self._rotate_face_ccw(U)
        tmp = self.state[F][0].copy()
        self.state[F][0] = self.state[L][0]
        self.state[L][0] = self.state[B][0]
        self.state[B][0] = self.state[R][0]
        self.state[R][0] = tmp

    def move_U2(self):
        self.move_U(); self.move_U()

    # D move  (bottom row = index 1 for 2x2)
    def move_D(self):
        self._rotate_face_cw(D)
        tmp = self.state[F][1].copy()
        self.state[F][1] = self.state[L][1]
        self.state[L][1] = self.state[B][1]
        self.state[B][1] = self.state[R][1]
        self.state[R][1] = tmp

    def move_Di(self):
        self._rotate_face_ccw(D)
        tmp = self.state[F][1].copy()
        self.state[F][1] = self.state[R][1]
        self.state[R][1] = self.state[B][1]
        self.state[B][1] = self.state[L][1]
        self.state[L][1] = tmp

    def move_D2(self):
        self.move_D(); self.move_D()

    # R move  (right column = index 1 for 2x2)
    def move_R(self):
        self._rotate_face_cw(R)
        tmp = self.state[U][:, 1].copy()
        self.state[U][:, 1] = self.state[F][:, 1]
        self.state[F][:, 1] = self.state[D][:, 1]
        self.state[D][:, 1] = self.state[B][::-1, 0]
        self.state[B][:, 0] = tmp[::-1]

    def move_Ri(self):
        self._rotate_face_ccw(R)
        tmp = self.state[U][:, 1].copy()
        self.state[U][:, 1] = self.state[B][::-1, 0]
        self.state[B][:, 0] = self.state[D][::-1, 1]
        self.state[D][:, 1] = self.state[F][:, 1]
        self.state[F][:, 1] = tmp

    def move_R2(self):
        self.move_R(); self.move_R()

    # L move  (B's right column = index 1 for 2x2)
    def move_L(self):
        self._rotate_face_cw(L)
        tmp = self.state[U][:, 0].copy()
        self.state[U][:, 0] = self.state[B][::-1, 1]
        self.state[B][:, 1] = self.state[D][::-1, 0]
        self.state[D][:, 0] = self.state[F][:, 0]
        self.state[F][:, 0] = tmp

    def move_Li(self):
        self._rotate_face_ccw(L)
        tmp = self.state[U][:, 0].copy()
        self.state[U][:, 0] = self.state[F][:, 0]
        self.state[F][:, 0] = self.state[D][:, 0]
        self.state[D][:, 0] = self.state[B][::-1, 1]
        self.state[B][:, 1] = tmp[::-1]

    def move_L2(self):
        self.move_L(); self.move_L()

    # F move  (U/L bottom/right = index 1 for 2x2)
    def move_F(self):
        self._rotate_face_cw(F)
        tmp = self.state[U][1].copy()
        self.state[U][1] = self.state[L][:, 1][::-1]
        self.state[L][:, 1] = self.state[D][0]
        self.state[D][0] = self.state[R][:, 0][::-1]
        self.state[R][:, 0] = tmp

    def move_Fi(self):
        self._rotate_face_ccw(F)
        tmp = self.state[U][1].copy()
        self.state[U][1] = self.state[R][:, 0]
        self.state[R][:, 0] = self.state[D][0][::-1]
        self.state[D][0] = self.state[L][:, 1]
        self.state[L][:, 1] = tmp[::-1]

    def move_F2(self):
        self.move_F(); self.move_F()

    # B move  (R/D right/bottom = index 1 for 2x2)
    def move_B(self):
        self._rotate_face_cw(B)
        tmp = self.state[U][0].copy()
        self.state[U][0] = self.state[R][:, 1]
        self.state[R][:, 1] = self.state[D][1][::-1]
        self.state[D][1] = self.state[L][:, 0]
        self.state[L][:, 0] = tmp[::-1]

    def move_Bi(self):
        self._rotate_face_ccw(B)
        tmp = self.state[U][0].copy()
        self.state[U][0] = self.state[L][:, 0][::-1]
        self.state[L][:, 0] = self.state[D][1]
        self.state[D][1] = self.state[R][:, 1][::-1]
        self.state[R][:, 1] = tmp

    def move_B2(self):
        self.move_B(); self.move_B()

    def apply_move(self, move_idx):
        _MOVE_DISPATCH[move_idx](self)

    MOVE_NAMES = [
        'U', "U'", 'U2',
        'D', "D'", 'D2',
        'R', "R'", 'R2',
        'L', "L'", 'L2',
        'F', "F'", 'F2',
        'B', "B'", 'B2',
    ]

    INVERSE_MOVES = [1, 0, 2, 4, 3, 5, 7, 6, 8, 10, 9, 11, 13, 12, 14, 16, 15, 17]

    def scramble(self, n):
        moves = []
        for _ in range(n):
            m = random.randint(0, 17)
            self.apply_move(m)
            moves.append(m)
        return moves

    def to_dict(self):
        return self.state.tolist()


_MOVE_DISPATCH = [
    Cube.move_U,  Cube.move_Ui, Cube.move_U2,
    Cube.move_D,  Cube.move_Di, Cube.move_D2,
    Cube.move_R,  Cube.move_Ri, Cube.move_R2,
    Cube.move_L,  Cube.move_Li, Cube.move_L2,
    Cube.move_F,  Cube.move_Fi, Cube.move_F2,
    Cube.move_B,  Cube.move_Bi, Cube.move_B2,
]
