import * as tf from '@tensorflow/tfjs';

export interface ModelConfigParams {
  inputNodes: number;
  outputNodes: number;
  hiddenLayers: number[];
  learningRate: number;
}

/**
 * Builds a TensorFlow sequential model based on the user's architecture.
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
    // Add dropout to prevent overfitting during self-play
    model.add(tf.layers.dropout({ rate: 0.1 }));
  }

  // Output Layer
  model.add(tf.layers.dense({
    units: params.outputNodes,
    activation: 'softmax' // We use softmax to get a probability distribution over moves
  }));

  const optimizer = tf.train.adam(params.learningRate);

  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return model;
}
