import tensorflow as tf
import numpy as np
import os
import random
import time
from logic import HexEngine, Encoder, rotate_coord

# ==========================================
# 🛑 AMD GPU SETUP (Linux / ROCm) 🛑
# To use your AMD GPU on Fedora, you usually need:
# pip install tensorflow-rocm
# ==========================================

print("GPUs Available: ", tf.config.list_physical_devices('GPU'))

# CONFIGURATION (Standardized 7x3500)
INPUT_NODES = 1372  
OUTPUT_NODES = 1356 
BATCH_SIZE = 128    
MAX_TURNS = 250
MEMORY_CAPACITY = 10000
PARALLEL_GAMES = 4  

REWARDS = {
    'win': 5.0,
    'draw': 0.5,
    'line3': 0.05,
    'line4': 0.15,
    'line5': 0.50,
    'block4': 0.20,
    'block5': 0.50,
    'eff': -0.005,
    'illegal': -0.05
}

def build_model():
    print("\n🧠 Building 7x3500 Dueling DQN (Keras 3)...")
    inputs = tf.keras.Input(shape=(INPUT_NODES,))
    x = inputs
    
    # Feature Extractor
    for i in range(7):
        x = tf.keras.layers.Dense(3500, activation='relu')(x)
    
    # Dueling Heads
    value = tf.keras.layers.Dense(1, activation='linear')(x)
    advantage = tf.keras.layers.Dense(OUTPUT_NODES, activation='linear')(x)
    
    # Keras 3 Math (Fixes the KerasTensor Error)
    def combine_duel(args):
        v, a = args
        return v + (a - tf.keras.ops.mean(a, axis=1, keepdims=True))

    q_values = tf.keras.layers.Lambda(combine_duel)([value, advantage])
    
    model = tf.keras.Model(inputs=inputs, outputs=q_values)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')
    return model

def get_best_move(model, state, engine, encoder, epsilon):
    if random.random() < epsilon:
        # Pick a move from the visible windows
        total_slots = encoder.input_size - 16
        idx = random.randint(0, total_slots - 1)
        q, r = encoder.decode_action(idx, engine.foci, encoder.radii)
        return q, r, idx
    
    # AI Prediction
    q_values = model.predict(state[np.newaxis, :], verbose=0)[0]
    sorted_indices = np.argsort(q_values)[::-1]
    
    for idx in sorted_indices:
        q, r = encoder.decode_action(idx, engine.foci, encoder.radii)
        if f"{q},{r}" not in engine.board:
            return q, r, int(idx)
    return 0, 0, 0

def play_game(model, encoder, epsilon):
    engine = HexEngine()
    history = []
    winner = None
    curr_player = 1
    
    while winner is None and engine.turn < MAX_TURNS:
        moves_to_make = 1 if (engine.turn == 0 and curr_player == 1) else 2
        turn_experiences = []
        
        for _ in range(moves_to_make):
            state = encoder.encode(engine, curr_player, MAX_TURNS)
            q, r, action_idx = get_best_move(model, state, engine, encoder, epsilon)
            
            # Tactical Reward Calculation
            t_bonus = REWARDS['eff']
            m_line = engine.get_max_line(q, r, curr_player)
            if m_line == 3: t_bonus += REWARDS['line3']
            elif m_line == 4: t_bonus += REWARDS['line4']
            elif m_line >= 5: t_bonus += REWARDS['line5']
            
            other_p = 2 if curr_player == 1 else 1
            e_line = engine.get_max_line(q, r, other_p)
            if e_line == 4: t_bonus += REWARDS['block4']
            elif e_line == 5: t_bonus += REWARDS['block5']
            
            # Move
            is_valid = engine.make_move(q, r, curr_player)
            if not is_valid: t_bonus += REWARDS['illegal']

            turn_experiences.append({
                'state': state,
                'action': action_idx,
                'player': curr_player,
                't_bonus': t_bonus
            })
            
            if engine.check_win(q, r, curr_player):
                winner = curr_player
                break
        
        history.extend(turn_experiences)
        curr_player = 2 if curr_player == 1 else 1
        engine.turn += 1
        
    return history, winner

def train():
    model = build_model()
    encoder = Encoder()
    memory = [] 
    
    print("\n🚀 Starting Self-Play Training Loop...")
    gen = 0
    epsilon = 0.2
    
    try:
        while True:
            # 1. Play Parallel Games
            for _ in range(PARALLEL_GAMES):
                game_hist, winner = play_game(model, encoder, epsilon)
                
                # 2. Rewards & Experience Collection
                for p in [1, 2]:
                    p_moves = [m for m in game_hist if m['player'] == p]
                    base_reward = 0.0
                    if winner == p: base_reward = REWARDS['win']
                    elif winner is not None: base_reward = -1.0
                    else: base_reward = REWARDS['draw']
                    
                    total_r = base_reward + sum(m['t_bonus'] for m in p_moves)
                    
                    for m in p_moves:
                        # Store simplified experiences
                        memory.append((m['state'], m['action'], total_r))
            
            if len(memory) > MEMORY_CAPACITY: memory = memory[-MEMORY_CAPACITY:]
            
            # 3. Train on Batch
            if len(memory) >= BATCH_SIZE:
                mini_batch = random.sample(memory, BATCH_SIZE)
                states = np.array([x[0] for x in mini_batch])
                targets = model.predict(states, verbose=0)
                for i in range(BATCH_SIZE):
                    targets[i][mini_batch[i][1]] = mini_batch[i][2]
                
                loss = model.train_on_batch(states, targets)
                gen += PARALLEL_GAMES
                print(f"Gen {gen} | Loss: {loss:.4f} | Eps: {epsilon:.2f} | RAM: {len(memory)}")
            
            # 4. Periodic Save & Decay
            if gen % 20 == 0:
                model.save('hex_model.keras')
                epsilon = max(0.05, epsilon * 0.99) 

    except KeyboardInterrupt:
        print("\nSaving and Exiting...")
        model.save('hex_model.keras')

if __name__ == "__main__":
    train()
