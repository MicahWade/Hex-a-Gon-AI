import numpy as np
import math
import random

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
        self.board = {} 
        self.history = []
        self.foci = [[0, 0] for _ in range(6)]
        self.turn = 0
        return self

    def get_max_line(self, q, r, player):
        search_dirs = [(1, 0), (0, 1), (1, -1)]
        max_len = 0
        for dq, dr in search_dirs:
            count = 1
            for i in range(1, 6):
                if self.board.get(coord_to_string(q + dq * i, r + dr * i)) == player:
                    count += 1
                else: break
            for i in range(1, 6):
                if self.board.get(coord_to_string(q - dq * i, r - dr * i)) == player:
                    count += 1
                else: break
            max_len = max(max_len, count)
        return max_len

    def get_tactical_move(self, player, block_chance=0.5):
        opponent = 2 if player == 1 else 1
        coords = []
        for k in self.board.keys():
            q, r = map(int, k.split(','))
            coords.append((q, r))
        
        if not coords: return 0, 0
        
        min_q = min([c[0] for c in coords]) - 2
        max_q = max([c[0] for c in coords]) + 2
        min_r = min([c[1] for c in coords]) - 2
        max_r = max([c[1] for c in coords]) + 2
        
        candidates = []
        for q in range(min_q, max_q + 1):
            for r in range(min_r, max_r + 1):
                key = coord_to_string(q, r)
                if key not in self.board:
                    score = 0
                    my_max = self.get_max_line(q, r, player)
                    enemy_max = self.get_max_line(q, r, opponent)
                    
                    if my_max >= 6: score += 1000
                    if enemy_max >= 5 and random.random() < block_chance: score += 500
                    if my_max == 5: score += 100
                    if my_max == 4: score += 50
                    if my_max == 3: score += 10
                    if enemy_max == 4 and random.random() < block_chance: score += 40
                    
                    score += random.random() * 2
                    candidates.append(((q, r), score))
        
        if not candidates: return 0, 0
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def make_move(self, q, r, player):
        key = coord_to_string(q, r)
        if key in self.board: return False
        self.board[key] = player
        self.history.append({'q': q, 'r': r, 'p': player})
        self.foci[0] = [q, r]
        return True

    def check_win(self, q, r, player):
        return self.get_max_line(q, r, player) >= 6

class Encoder:
    @staticmethod
    def get_focal_indices(radius):
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
        self.input_size = sum(len(w) for w in self.windows) + 16

    def encode(self, engine, current_player, max_turns):
        input_vec = []
        for i, window in enumerate(self.windows):
            focus_q, focus_r = engine.foci[i]
            for dq, dr in window:
                val = engine.board.get(coord_to_string(focus_q + dq, focus_r + dr))
                if val is None: input_vec.append(0)
                elif val == current_player: input_vec.append(1)
                else: input_vec.append(-1)
        for f_q, f_r in engine.foci:
            input_vec.append(f_q / 100.0)
            input_vec.append(f_r / 100.0)
        input_vec.append(1.0 if current_player == 1 else 0.0)
        input_vec.append(1.0)
        input_vec.append(0.0)
        input_vec.append(engine.turn / max_turns)
        return np.array(input_vec, dtype=np.float32)

    def decode_action(self, index, foci, radii):
        curr_idx = 0
        for i, window in enumerate(self.windows):
            if index < curr_idx + len(window):
                dq, dr = window[index - curr_idx]
                f_q, f_r = foci[i]
                return f_q + dq, f_r + dr
            curr_idx += len(window)
        return foci[0][0], foci[0][1]
