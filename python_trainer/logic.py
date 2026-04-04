import numpy as np
import math

# Axial Hex Constants
SQRT3 = math.sqrt(3)

def coord_to_string(q, r):
    return f"{q},{r}"

def rotate_coord(q, r, times):
    for _ in range(times % 6):
        new_q = -r
        new_r = q + r
        q, r = new_q, new_r
    return q, r

class HexEngine:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = {} # Map string "q,r" -> player ID
        self.history = []
        self.foci = [[0, 0] for _ in range(6)]
        self.turn = 0
        return self

    def get_max_line(self, q, r, player):
        search_dirs = [(1, 0), (0, 1), (1, -1)]
        max_len = 0
        for dq, dr in search_dirs:
            count = 1
            # Forward
            for i in range(1, 6):
                if self.board.get(coord_to_string(q + dq * i, r + dr * i)) == player:
                    count += 1
                else: break
            # Backward
            for i in range(1, 6):
                if self.board.get(coord_to_string(q - dq * i, r - dr * i)) == player:
                    count += 1
                else: break
            max_len = max(max_len, count)
        return max_len

    def make_move(self, q, r, player):
        key = coord_to_string(q, r)
        if key in self.board: return False
        self.board[key] = player
        self.history.append({'q': q, 'r': r, 'p': player})
        # Update Focal Points (Simplified: move most recent to index 0)
        self.foci[0] = [q, r]
        return True

    def check_win(self, q, r, player):
        return self.get_max_line(q, r, player) >= 6

class Encoder:
    @staticmethod
    def get_focal_indices(radius):
        """Pre-calculates axial coordinates for a given focal radius (Spiral order)"""
        coords = []
        for q in range(-radius, radius + 1):
            for r in range(max(-radius, -q-radius), min(radius, -q+radius) + 1):
                coords.append((q, r))
        return coords

    def __init__(self, radii={'global': 14, 'self': 8, 'memory': 6}):
        self.radii = radii
        self.windows = [
            self.get_focal_indices(radii['global']),
            self.get_focal_indices(radii['self']),
            self.get_focal_indices(radii['memory']),
            self.get_focal_indices(radii['memory']),
            self.get_focal_indices(radii['memory']),
            self.get_focal_indices(radii['memory'])
        ]
        self.input_size = sum(len(w) for w in self.windows) + 16 # Hexes + 16 Metadata nodes

    def encode(self, engine, current_player, max_turns):
        input_vec = []
        
        # 1. Focal Window Hexes
        for i, window in enumerate(self.windows):
            focus_q, focus_r = engine.foci[i]
            for dq, dr in window:
                val = engine.board.get(coord_to_string(focus_q + dq, focus_r + dr))
                if val is None: input_vec.append(0)
                elif val == current_player: input_vec.append(1)
                else: input_vec.append(-1)

        # 2. Localization Nodes (12 nodes)
        for f_q, f_r in engine.foci:
            input_vec.append(f_q / 100.0)
            input_vec.append(f_r / 100.0)

        # 3. Context Nodes (4 nodes)
        input_vec.append(1.0 if current_player == 1 else 0.0)
        input_vec.append(1.0) # Bias
        input_vec.append(0.0) # Ground
        input_vec.append(engine.turn / max_turns)

        return np.array(input_vec, dtype=np.float32)

    def decode_action(self, index, foci, radii):
        # Maps index back to absolute Q, R
        curr_idx = 0
        for i, window in enumerate(self.windows):
            if index < curr_idx + len(window):
                dq, dr = window[index - curr_idx]
                f_q, f_r = foci[i]
                return f_q + dq, f_r + dr
            curr_idx += len(window)
        return foci[0][0], foci[0][1]
