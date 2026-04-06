import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
import multiprocessing as mp
from logic import HexEngine, Encoder

# ==========================================
# 🚀 PyTorch Training Factory v3.0 (Hyper-Utilization)
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training on Device: {device}")

# CONFIGURATION
INPUT_NODES = 1372
OUTPUT_NODES = 1356
BATCH_SIZE = 512     # High batch size pushes GPU utilization
MEMORY_CAPACITY = 20000
MAX_TURNS = 250

# MULTIPROCESSING CONFIG
NUM_WORKERS = max(1, mp.cpu_count() - 2) 

# ADAPTIVE EPSILON
EPSILON_BUCKETS = [0.05, 0.15, 0.35, 0.65]

REWARDS = {
    'win': 5.0, 'draw': 0.5, 'line3': 0.05, 'line4': 0.15,
    'line5': 0.50, 'block4': 0.20, 'block5': 0.50, 'eff': -0.005, 'illegal': -0.05
}

class DuelingDQN(nn.Module):
    def __init__(self):
        super(DuelingDQN, self).__init__()
        layers = []
        last_dim = INPUT_NODES
        for _ in range(7):
            layers.append(nn.Linear(last_dim, 3500))
            layers.append(nn.ReLU())
            last_dim = 3500
        self.feature_extractor = nn.Sequential(*layers)
        self.value_stream = nn.Linear(3500, 1)
        self.advantage_stream = nn.Linear(3500, OUTPUT_NODES)

    def forward(self, x):
        features = self.feature_extractor(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return value + (advantages - advantages.mean(dim=1, keepdim=True))

def get_best_move_cpu(model, state, engine, encoder, epsilon):
    if random.random() < epsilon:
        idx = random.randint(0, OUTPUT_NODES - 1)
        q, r = encoder.decode_action(idx, engine.foci, encoder.radii)
        return q, r, idx
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0)
        q_values = model(state_t).numpy()[0]
        sorted_indices = np.argsort(q_values)[::-1]
        for idx in sorted_indices:
            q, r = encoder.decode_action(idx, engine.foci, encoder.radii)
            if f"{q},{r}" not in engine.board:
                return q, r, int(idx)
    return 0, 0, 0

def worker_play_game(model_state, epsilon, bot_type, bot_id):
    model = DuelingDQN()
    model.load_state_dict(model_state)
    model.eval()
    encoder = Encoder()
    engine = HexEngine()
    history = []
    winner = None
    curr_player = 1
    while winner is None and engine.turn < MAX_TURNS:
        moves_to_make = 1 if (engine.turn == 0 and curr_player == 1) else 2
        for _ in range(moves_to_make):
            if curr_player == bot_id:
                if bot_type == 'RANDOM':
                    q, r = random.randint(-10, 10), random.randint(-10, 10)
                    while f"{q},{r}" in engine.board: q, r = random.randint(-10, 10), random.randint(-10, 10)
                    action_idx = 0
                else:
                    q, r = engine.get_tactical_move(curr_player, 0.5)
                    action_idx = 0
            else:
                state = encoder.encode(engine, curr_player, MAX_TURNS)
                q, r, action_idx = get_best_move_cpu(model, state, engine, encoder, epsilon)
            t_bonus = REWARDS['eff']
            m_line = engine.get_max_line(q, r, curr_player)
            if m_line == 3: t_bonus += REWARDS['line3']
            elif m_line == 4: t_bonus += REWARDS['line4']
            elif m_line >= 5: t_bonus += REWARDS['line5']
            other_p = 2 if curr_player == 1 else 1
            e_line = engine.get_max_line(q, r, other_p)
            if e_line == 4: t_bonus += REWARDS['block4']
            elif e_line == 5: t_bonus += REWARDS['block5']
            state_before = encoder.encode(engine, curr_player, MAX_TURNS)
            is_valid = engine.make_move(q, r, curr_player)
            if not is_valid: t_bonus += REWARDS['illegal']
            if curr_player != bot_id:
                history.append({'state': state_before, 'action': action_idx, 'player': curr_player, 't_bonus': t_bonus})
            if engine.check_win(q, r, curr_player):
                winner = curr_player
                break
        curr_player = 2 if curr_player == 1 else 1
        engine.turn += 1
    return history, winner

def train():
    mp.set_start_method('spawn', force=True)
    model = DuelingDQN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    gen = 0
    bucket_stats = [1, 1, 1, 1]
    bot_win_rate = 1.0
    if os.path.exists('hex_brain.pt'):
        print("📂 Loading saved checkpoint...")
        checkpoint = torch.load('hex_brain.pt', map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            try: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except: pass
            gen = checkpoint.get('gen', 0)
            bucket_stats = checkpoint.get('bucket_stats', [1, 1, 1, 1])
            bot_win_rate = checkpoint.get('bot_win_rate', 1.0)
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ Resumed at Gen {gen}")
    loss_fn = nn.MSELoss()
    memory = []
    print(f"\n🔥 Launching {NUM_WORKERS} Parallel Workers...")
    try:
        pool = mp.Pool(processes=NUM_WORKERS)
        while True:
            current_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            current_bot_freq = 0.025 + (bot_win_rate * (0.20 - 0.025))
            tasks = []
            for _ in range(NUM_WORKERS):
                raw_total = sum(bucket_stats)
                min_w = raw_total * 0.10
                floored_stats = [max(s, min_w) for s in bucket_stats]
                weights = [s/sum(floored_stats) for s in floored_stats]
                bucket_idx = np.random.choice(len(EPSILON_BUCKETS), p=weights)
                rand_val = random.random()
                is_bot_game = rand_val < current_bot_freq
                bot_type = random.choice(['RANDOM', 'TACTICAL']) if is_bot_game else 'NONE'
                bot_id = random.choice([1, 2]) if is_bot_game else 0
                tasks.append((current_model_state, EPSILON_BUCKETS[bucket_idx], bot_type, bot_id))
            results = pool.starmap(worker_play_game, tasks)
            for game_hist, winner in results:
                for p in [1, 2]:
                    p_moves = [m for m in game_hist if m['player'] == p]
                    if not p_moves: continue
                    base = REWARDS['win'] if winner == p else (-1.0 if winner is not None else REWARDS['draw'])
                    total_r = base + sum(m['t_bonus'] for m in p_moves)
                    for m in p_moves:
                        memory.append((m['state'], m['action'], total_r))
            if len(memory) > MEMORY_CAPACITY: memory = memory[-MEMORY_CAPACITY:]
            if len(memory) >= BATCH_SIZE:
                mini_batch = random.sample(memory, BATCH_SIZE)
                states = torch.FloatTensor(np.array([x[0] for x in mini_batch])).to(device)
                actions = torch.LongTensor(np.array([x[1] for x in mini_batch])).to(device)
                rewards = torch.FloatTensor(np.array([x[2] for x in mini_batch])).to(device)
                current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                loss = loss_fn(current_q, rewards)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                gen += NUM_WORKERS
                print(f"Gen {gen} | Loss: {loss.item():.4f} | System Load: HIGH | RAM: {len(memory)}")
            if gen % 50 == 0:
                torch.save({'gen': gen, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'bucket_stats': bucket_stats, 'bot_win_rate': bot_win_rate}, 'hex_brain.pt')
    except KeyboardInterrupt:
        print(f"\n💾 Saving Progress at Gen {gen}...")
        torch.save({'gen': gen, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'bucket_stats': bucket_stats, 'bot_win_rate': bot_win_rate}, 'hex_brain.pt')
    finally:
        pool.close(); pool.join()

if __name__ == "__main__":
    train()
