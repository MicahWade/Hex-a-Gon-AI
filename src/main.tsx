import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import * as tf from '@tensorflow/tfjs';

// PERFORMANCE: Try to use WebGPU for maximum training speed
// Falls back to WebGL if WebGPU is not supported by the browser
const initTf = async () => {
  try {
    // Check for WebGPU support
    if ('gpu' in navigator) {
      await tf.setBackend('webgpu');
      console.log('Using WebGPU backend');
    } else {
      await tf.setBackend('webgl');
      console.log('Using WebGL backend');
    }
  } catch (e) {
    await tf.setBackend('cpu');
    console.log('Falling back to CPU backend');
  }
  await tf.ready();
};

initTf().then(() => {
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <App />
    </StrictMode>,
  )
});
