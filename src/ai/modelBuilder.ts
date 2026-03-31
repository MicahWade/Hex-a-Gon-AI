import * as tf from '@tensorflow/tfjs';

export interface ModelConfigParams {
  inputNodes: number;
  outputNodes: number;
  hiddenLayers: number[];
  learningRate: number;
}

/**
 * Builds a robust Deep Q-Network.
 * Standard Sequential model for maximum stability.
 */
export function createModel(params: ModelConfigParams): tf.LayersModel {
  const model = tf.sequential();

  // Input Layer
  model.add(tf.layers.dense({
    units: params.hiddenLayers[0],
    inputShape: [params.inputNodes],
    activation: 'relu',
    kernelInitializer: 'leCunNormal'
  }));

  // Hidden Layers
  for (let i = 1; i < params.hiddenLayers.length; i++) {
    model.add(tf.layers.dense({
      units: params.hiddenLayers[i],
      activation: 'relu',
      kernelInitializer: 'leCunNormal'
    }));
    // Dropout helps prevent overfitting in complex games
    model.add(tf.layers.dropout({ rate: 0.1 }));
  }

  // Output Layer: Q-values for every possible move
  model.add(tf.layers.dense({
    units: params.outputNodes,
    activation: 'linear'
  }));

  const optimizer = tf.train.adam(params.learningRate);

  model.compile({
    optimizer: optimizer,
    loss: 'meanSquaredError',
    metrics: ['accuracy']
  });

  return model;
}
