import tensorflow as tf
import numpy as np
import os
import json

# ==========================================
# 🛑 AMD GPU SETUP & WARNING 🛑
# ==========================================
# Standard `tensorflow` defaults to Nvidia CUDA.
# To ensure this maxes out your AMD GPU, you must install the specific AMD version:
# 
# 👉 For Windows: pip install tensorflow-directml-plugin
# 👉 For Linux: pip install tensorflow-rocm
# 👉 For Mac (M1/M2/M3): pip install tensorflow-metal
# ==========================================

print("GPUs Available: ", tf.config.list_physical_devices('GPU'))
if len(tf.config.list_physical_devices('GPU')) == 0:
    print("WARNING: No GPU detected! Training on CPU will be very slow.")

INPUT_NODES = 1372  # Matches Multi-Focal Vision setup
OUTPUT_NODES = 1356 # Matches move choices

# HYPERPARAMETERS
BATCH_SIZE = 256    

def build_massive_model():
    """
    Builds the high-performance 7 layers of 3500 neurons.
    """
    print("\n🧠 Building Massive 7x3500 Dueling DQN Model...")
    inputs = tf.keras.Input(shape=(INPUT_NODES,))
    x = inputs
    for i in range(7):
        x = tf.keras.layers.Dense(3500, activation='relu', name=f'dense_{i}')(x)
    value = tf.keras.layers.Dense(1, activation='linear', name='value')(x)
    advantage = tf.keras.layers.Dense(OUTPUT_NODES, activation='linear', name='advantage')(x)
    adv_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)
    q_values = value + (advantage - adv_mean)
    model = tf.keras.Model(inputs=inputs, outputs=q_values)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')
    return model

def load_memory_from_json():
    """
    Looks for memory.json exported from the browser.
    """
    memory_files = [f for f in os.listdir('.') if f.startswith('memory_') and f.endswith('.json')]
    if not memory_files:
        print("❌ No 'memory_*.json' found. Please export training data from the website and place it here.")
        return None, None
    
    latest_file = sorted(memory_files)[-1]
    print(f"📂 Loading training data from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    states = np.array([exp['state'] for exp in data]).astype(np.float32)
    rewards = np.array([exp['reward'] for exp in data]).astype(np.float32)
    actions = np.array([exp['action'] for exp in data])
    
    # Create Q-Targets (simplified for offline training)
    # On a real DQN we would use the Bellman equation, but here we'll 
    # train the model to output the reward for the selected action.
    targets = np.zeros((len(data), OUTPUT_NODES)).astype(np.float32)
    for i in range(len(actions)):
        if actions[i] < OUTPUT_NODES:
            targets[i][actions[i]] = rewards[i]
            
    return states, targets

def train_loop():
    model = build_massive_model()
    model.summary()

    states, targets = load_memory_from_json()
    
    if states is not None:
        print(f"🚀 Pushing {len(states)} real experiences to GPU...")
        dataset = tf.data.Dataset.from_tensor_slices((states, targets))
        dataset = dataset.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        try:
            for epoch in range(1, 101):
                print(f"\n--- Epoch {epoch} ---")
                model.fit(dataset, epochs=1, verbose=1)
                
                if epoch % 10 == 0:
                    model.save('hex_model.keras')
                    print("\n💾 Model saved! Run converter to use in browser.")
        except KeyboardInterrupt:
            print("\nStopping...")
            model.save('hex_model.keras')
    else:
        print("Waiting for data. Training aborted.")

if __name__ == "__main__":
    train_loop()
