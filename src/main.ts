import './style.css';
import { FarmSimulation } from './simulation';
import { FarmRenderer } from './world';

const canvas = document.querySelector<HTMLCanvasElement>('#farm-canvas')!;
const simulation = new FarmSimulation();
const world = new FarmRenderer(canvas, simulation);
let previous = performance.now();
let simAccumulator = 0;

function frame(now: number) {
  const delta = Math.min(0.1, (now - previous) / 1000);
  previous = now;
  simAccumulator += delta;
  // The ecological clock stays independent of the render rate.
  while (simAccumulator >= 0.5) {
    simulation.step(0.25);
    simAccumulator -= 0.5;
  }
  world.draw(now);
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
