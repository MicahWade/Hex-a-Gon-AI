import tensorflow as tf
import numpy as np
import os
import random
import time
from logic import HexEngine, Encoder, rotate_coord

# ==========================================
# 🛑 AMD GPU SETUP 🛑
# Ensure you installed: pip install tensorflow-directml-plugin (Windows)
# ==========================================

print("GPUs Available: ", tf.config.list_physical_devices('GPU'))

# CONFIGURATION (Mirrors Website)
INPUT_NODES = 1372  
OUTPUT_NODES = 1356 
BATCH_SIZE = 128    # Adjusted for massive model stability
MAX_TURNS = 250
MEMORY_CAPACITY = 5000
PARALLEL_GAMES = 8  # How many games to play before each training batch

# REWARDS
REWARDS = {
    'win': 5.0,
    'draw': 0.5,
    'line3': 0.05,
    'line4': 0.15,
    'line5': 0.50,
    'block4': 0.20,
    'block5': 0.50,
    'eff': -0.005
}

def build_model():
    print("\n🧠 Building 7x3500 Dueling DQN...")
    inputs = tf.keras.Input(shape=(INPUT_NODES,))
    x = inputs
    for i in range(7):
        x = tf.keras.layers.Dense(3500, activation='relu')(x)
    value = tf.keras.layers.Dense(1, activation='linear')(x)
    advantage = tf.keras.layers.Dense(OUTPUT_NODES, activation='linear')(x)
    adv_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)
    q_values = value + (advantage - adv_mean)
    model = tf.keras.Model(inputs=inputs, outputs=q_values)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')
    return model

def get_best_move(model, state, engine, encoder, player, epsilon):
    if random.random() < epsilon:
        # Random valid move
        q, r = random.randint(-10, 10), random.randint(-10, 10)
        while f"{q},{r}" in engine.board:
            q, r = random.randint(-10, 10), random.randint(-10, 10)
        return q, r
    
    # AI Prediction
    q_values = model.predict(state[np.newaxis, :], verbose=0)[0]
    # Sort indices by probability
    sorted_indices = np.argsort(q_values)[::-1]
    
    for idx in sorted_indices:
        q, r = encoder.decode_action(idx, engine.foci, encoder.radii)
        if f"{q},{r}" not in engine.board:
            return q, r
    return 0, 0 # Fallback

def play_game(model, encoder, epsilon):
    engine = HexEngine()
    history = []
    winner = None
    curr_player = 1
    
    while winner is None and engine.turn < MAX_TURNS:
        # P1 first move is always (0,0) per rules
        if engine.turn == 0 and curr_player == 1:
            moves_to_make = 1
        else:
            moves_to_make = 2
            
        turn_experiences = []
        
        for _ in range(moves_to_make):
            state = encoder.encode(engine, curr_player, MAX_TURNS)
            q, r = get_best_move(model, state, engine, encoder, curr_player, epsilon)
            
            # Tactical Bonuses
            t_bonus = REWARDS['eff']
            m_line = engine.get_max_line(q, r, curr_player)
            if m_line == 3: t_bonus += REWARDS['line3']
            elif m_line == 4: t_bonus += REWARDS['line4']
            elif m_line == 5: t_bonus += REWARDS['line5']
            
            other_p = 2 if curr_player == 1 else 1
            e_line = engine.get_max_line(q, r, other_p)
            if e_line == 4: t_bonus += REWARDS['block4']
            elif e_line == 5: t_bonus += REWARDS['block5']
            
            # Execute
            engine.make_move(q, r, curr_player)
            
            # Store experience for this move
            turn_experiences.append({
                'state': state,
                'action': q, # Need to map Q,R back to index for training
                'coord': (q, r),
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
    memory = [] # Buffer of (state, action_idx, final_reward)
    
    print("\n🚀 Starting Self-Play Loop...")
    gen = 0
    epsilon = 0.2
    
    try:
        while True:
            batch_experiences = []
            
            # 1. Play Parallel Games
            for _ in range(PARALLEL_GAMES):
                game_hist, winner = play_game(model, encoder, epsilon)
                
                # 2. Process Final Rewards (End of Game)
                for p in [1, 2]:
                    p_moves = [m for m in game_hist if m['player'] == p]
                    base = REWARDS['win'] if winner == p else (-1.0 if winner is not None else REWARDS['draw'])
                    total_r = base + sum(m['t_bonus'] for m in p_moves)
                    
                    # 3. Data Augmentation (6x rotations)
                    for m in p_moves:
                        for r in range(6):
                            # Map move to output index
                            # Note: To be fully accurate we need to rotate the board state too
                            # for now we'll store the base experience
                            batch_experiences.append((m['state'], 0, total_r)) # Simplified index for now
            
            memory.extend(batch_experiences)
            if len(memory) > MEMORY_CAPACITY: memory = memory[-MEMORY_CAPACITY:]
            
            # 4. Train on Memory
            if len(memory) >= BATCH_SIZE:
                mini_batch = random.sample(memory, BATCH_SIZE)
                states = np.array([x[0] for x in mini_batch])
                targets = model.predict(states, verbose=0)
                for i in range(BATCH_SIZE):
                    targets[i][mini_batch[i][1]] = mini_batch[i][2] # Update Q-target
                
                loss = model.train_on_batch(states, targets)
                gen += PARALLEL_GAMES
                print(f"Gen {gen} | Memory: {len(memory)} | Loss: {loss:.4f} | Eps: {epsilon:.2f}")
            
            if gen % 100 == 0:
                model.save('hex_model.keras')
                epsilon = max(0.05, epsilon * 0.995) # Decay randomness

    except KeyboardInterrupt:
        print("\nSaving and Exiting...")
        model.save('hex_model.keras')

if __name__ == "__main__":
    train()
