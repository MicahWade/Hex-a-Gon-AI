import * as tf from '@tensorflow/tfjs';

export interface ModelConfigParams {
  inputNodes: number;
  outputNodes: number;
  hiddenLayers: number[];
  learningRate: number;
}

/**
 * Builds a Dueling Deep Q-Network (Dueling DQN).
 * It splits the network into two streams: Value and Advantage.
 */
export function createModel(params: ModelConfigParams): tf.LayersModel {
  const input = tf.input({ shape: [params.inputNodes] });

  // 1. Shared Feature Extractor (Hidden Layers)
  let x = input;
  for (let i = 0; i < params.hiddenLayers.length; i++) {
    x = tf.layers.dense({
      units: params.hiddenLayers[i],
      activation: 'relu',
      kernelInitializer: 'leCunNormal'
    }).apply(x) as tf.SymbolicTensor;
    
    x = tf.layers.dropout({ rate: 0.1 }).apply(x) as tf.SymbolicTensor;
  }

  // 2. Dueling Streams
  // Value Stream: Estimating the value of the state itself (V(s))
  const value = tf.layers.dense({
    units: 1,
    activation: 'linear',
    name: 'value_head'
  }).apply(x) as tf.SymbolicTensor;

  // Advantage Stream: Estimating the advantage of each action (A(s, a))
  const advantage = tf.layers.dense({
    units: params.outputNodes,
    activation: 'linear',
    name: 'advantage_head'
  }).apply(x) as tf.SymbolicTensor;

  // 3. Combine Streams: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
  const advantageMean = tf.layers.lambda({
    func: (t: tf.Tensor) => t.mean(1, true),
    name: 'advantage_mean'
  }).apply(advantage) as tf.SymbolicTensor;

  const advantageCentered = tf.layers.lambda({
    func: (t: tf.Tensor[]) => t[0].sub(t[1]),
    name: 'advantage_centered'
  }).apply([advantage, advantageMean]) as tf.SymbolicTensor;

  const qValues = tf.layers.lambda({
    func: (t: tf.Tensor[]) => t[0].add(t[1]),
    name: 'q_values'
  }).apply([value, advantageCentered]) as tf.SymbolicTensor;

  const model = tf.model({ inputs: input, outputs: qValues });

  const optimizer = tf.train.adam(params.learningRate);

  model.compile({
    optimizer: optimizer,
    loss: 'meanSquaredError', // DQN typically uses MSE or Huber loss
    metrics: ['accuracy']
  });

  return model;
}
