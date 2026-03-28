import * as tf from '@tensorflow/tfjs';

export interface ModelMetadata {
  name: string;
  timestamp: number;
  inputNodes: number;
  outputNodes: number;
  hiddenLayers: number[];
  focalRadii: { global: number; self: number; memory: number };
  generation: number;
  maxTurns: number;
  batchSize: number;
  epsilon: number;
}

const METADATA_KEY = 'hexagon-model-vault-metadata';

export async function saveModelToVault(model: tf.LayersModel, metadata: ModelMetadata): Promise<void> {
  const vault = getVaultMetadata();
  const index = vault.findIndex(m => m.name === metadata.name);
  if (index !== -1) {
    vault[index] = metadata;
  } else {
    vault.push(metadata);
  }
  
  await model.save(`indexeddb://${metadata.name}`);
  localStorage.setItem(METADATA_KEY, JSON.stringify(vault));
}

export function getVaultMetadata(): ModelMetadata[] {
  const data = localStorage.getItem(METADATA_KEY);
  return data ? JSON.parse(data) : [];
}

export async function deleteModelFromVault(name: string): Promise<void> {
  let vault = getVaultMetadata();
  vault = vault.filter(m => m.name !== name);
  localStorage.setItem(METADATA_KEY, JSON.stringify(vault));
  await tf.io.removeModel(`indexeddb://${name}`);
}

export async function loadModelFromVault(name: string): Promise<tf.LayersModel> {
  return await tf.loadLayersModel(`indexeddb://${name}`);
}
