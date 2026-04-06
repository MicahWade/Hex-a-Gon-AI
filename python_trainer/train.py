import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
from logic import HexEngine, Encoder

# ==========================================
# 🛑 AMD GPU SETUP 🛑
# pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training on Device: {device}")

# CONFIGURATION
INPUT_NODES = 1372
OUTPUT_NODES = 1356
BATCH_SIZE = 128
MAX_TURNS = 250
MEMORY_CAPACITY = 10000
PARALLEL_GAMES = 4

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

def get_best_move(model, state, engine, encoder, epsilon):
    if random.random() < epsilon:
        idx = random.randint(0, OUTPUT_NODES - 1)
        q, r = encoder.decode_action(idx, engine.foci, encoder.radii)
        return q, r, idx
    with torch.no_grad():
        state_t = torch.FloatTensor(state).to(device).unsqueeze(0)
        q_values = model(state_t).cpu().numpy()[0]
        sorted_indices = np.argsort(q_values)[::-1]
        for idx in sorted_indices:
            q, r = encoder.decode_action(idx, engine.foci, encoder.radii)
            if f"{q},{r}" not in engine.board:
                return q, r, int(idx)
    return 0, 0, 0

def play_game(model, encoder, epsilon, bot_type='NONE', bot_player_id=0):
    engine = HexEngine()
    history = []
    winner = None
    curr_player = 1
    
    while winner is None and engine.turn < MAX_TURNS:
        moves_to_make = 1 if (engine.turn == 0 and curr_player == 1) else 2
        for _ in range(moves_to_make):
            # Bot Move Logic
            if curr_player == bot_player_id:
                if bot_type == 'RANDOM':
                    q, r = random.randint(-10, 10), random.randint(-10, 10)
                    while f"{q},{r}" in engine.board:
                        q, r = random.randint(-10, 10), random.randint(-10, 10)
                    # We need an action index for training (approximate)
                    action_idx = 0 
                else: # TACTICAL
                    q, r = engine.get_tactical_move(curr_player, 0.5)
                    action_idx = 0
            else:
                state = encoder.encode(engine, curr_player, MAX_TURNS)
                q, r, action_idx = get_best_move(model, state, engine, encoder, epsilon)
            
            t_bonus = REWARDS['eff']
            m_line = engine.get_max_line(q, r, curr_player)
            if m_line == 3: t_bonus += REWARDS['line3']
            elif m_line == 4: t_bonus += REWARDS['line4']
            elif m_line >= 5: t_bonus += REWARDS['line5']
            
            other_p = 2 if curr_player == 1 else 1
            e_line = engine.get_max_line(q, r, other_p)
            if e_line == 4: t_bonus += REWARDS['block4']
            elif e_line == 5: t_bonus += REWARDS['block5']
            
            # Record state BEFORE the move
            state_before = encoder.encode(engine, curr_player, MAX_TURNS)
            
            is_valid = engine.make_move(q, r, curr_player)
            if not is_valid: t_bonus += REWARDS['illegal']

            # Only record experiences for the main model (not the bots)
            if curr_player != bot_player_id:
                history.append({'state': state_before, 'action': action_idx, 'player': curr_player, 't_bonus': t_bonus})
            
            if engine.check_win(q, r, curr_player):
                winner = curr_player
                break
        curr_player = 2 if curr_player == 1 else 1
        engine.turn += 1
    return history, winner

def train():
    model = DuelingDQN().to(device)
    
    # RESUME LOGIC
    if os.path.exists('hex_brain.pt'):
        print("📂 Found existing model 'hex_brain.pt'. Resuming training...")
        model.load_state_dict(torch.load('hex_brain.pt', map_location=device))
    else:
        print("🆕 No saved model found. Initializing new brain.")

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()
    encoder = Encoder()
    memory = []
    
    print("\n🚀 Starting PyTorch Factory Loop (70/15/15 Split)...")
    gen = 0
    epsilon = 0.2
    
    try:
        while True:
            for _ in range(PARALLEL_GAMES):
                # Choose Opponent: 70% Self, 15% Random, 15% Tactical
                rand_val = random.random()
                bot_type = 'RANDOM' if rand_val < 0.15 else ('TACTICAL' if rand_val < 0.30 else 'NONE')
                bot_id = random.choice([1, 2]) if bot_type != 'NONE' else 0
                
                game_hist, winner = play_game(model, encoder, epsilon, bot_type, bot_id)
                
                # Rewards
                for p in [1, 2]:
                    if p == bot_id: continue # Don't learn from bot perspectives
                    p_moves = [m for m in game_hist if m['player'] == p]
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
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                gen += PARALLEL_GAMES
                if gen % 4 == 0:
                    print(f"Gen {gen} | Loss: {loss.item():.4f} | Eps: {epsilon:.2f} | RAM: {len(memory)}")
            
            if gen % 100 == 0:
                torch.save(model.state_dict(), 'hex_brain.pt')
                epsilon = max(0.05, epsilon * 0.995)

    except KeyboardInterrupt:
        print("\nSaving and Exiting...")
        torch.save(model.state_dict(), 'hex_brain.pt')

if __name__ == "__main__":
    train()
