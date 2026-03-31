import * as tf from '@tensorflow/tfjs';

export interface ModelConfigParams {
  inputNodes: number;
  outputNodes: number;
  hiddenLayers: number[];
  learningRate: number;
}

/**
 * Builds a Deep Q-Network.
 * (Simplified for maximum compatibility)
 */
export function createModel(params: ModelConfigParams): tf.LayersModel {
  const model = tf.sequential();

  // 1. Shared Feature Extractor
  model.add(tf.layers.dense({
    units: params.hiddenLayers[0],
    inputShape: [params.inputNodes],
    activation: 'relu',
    kernelInitializer: 'leCunNormal'
  }));
  model.add(tf.layers.dropout({ rate: 0.1 }));

  for (let i = 1; i < params.hiddenLayers.length; i++) {
    model.add(tf.layers.dense({
      units: params.hiddenLayers[i],
      activation: 'relu',
      kernelInitializer: 'leCunNormal'
    }));
    model.add(tf.layers.dropout({ rate: 0.1 }));
  }

  // Output Layer: Q-values for each move
  model.add(tf.layers.dense({
    units: params.outputNodes,
    activation: 'linear' // Q-learning expects raw values
  }));

  const optimizer = tf.train.adam(params.learningRate);

  model.compile({
    optimizer: optimizer,
    loss: 'meanSquaredError',
    metrics: ['accuracy']
  });

  return model;
}
