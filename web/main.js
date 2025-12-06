/**
 * Entry point wiring together input, world streaming, and rendering.
 * @module main
 */
import { INTERACT_RANGE, PLAYER_EYE, TICKS_PER_SEC } from './config.js';
import { InputManager } from './input.js';
import { Player } from './player.js';
import { Renderer } from './renderer.js';
import { UIOverlay } from './ui.js';
import { World } from './world.js';
import { BLOCKS } from './blocks.js';
import { raycast } from './util.js';

/**
 * Entry point for the browser client.
 */
class App {
  constructor() {
    this.canvas = /** @type {HTMLCanvasElement} */ (document.getElementById('gl-canvas'));
    this.uiCanvas = /** @type {HTMLCanvasElement} */ (document.getElementById('ui-canvas'));
    this.renderer = new Renderer(this.canvas);
    this.ui = new UIOverlay(this.uiCanvas);
    this.input = new InputManager(this.canvas);
    this.player = new Player();
    this.world = new World();
    this.hotbar = BLOCKS.slice(0, 9);
    this.selectedBlock = this.hotbar[0];
    this.lastTime = performance.now();
    this.accumulator = 0;
    window.addEventListener('resize', () => this.handleResize());
    this.handleResize();
    this.ready = false;
    this.bootstrap();
  }

  async bootstrap() {
    await this.renderer.loadAtlas();
    this.ready = true;
    requestAnimationFrame((t) => this.loop(t));
  }

  handleResize() {
    const width = window.innerWidth;
    const height = window.innerHeight;
    this.canvas.width = width;
    this.canvas.height = height;
    this.ui.resize(width, height);
    this.renderer.resize(width, height);
  }

  loop(timestamp) {
    if (!this.ready) return;
    const dt = (timestamp - this.lastTime) / 1000;
    this.lastTime = timestamp;
    this.accumulator += dt;
    const tick = 1 / TICKS_PER_SEC;
    const control = this.input.sample();
    control.lookYaw = this.input.look[0];
    control.lookPitch = this.input.look[1];
    this.selectedBlock = this.hotbar[control.hotbar] ?? this.selectedBlock;

    while (this.accumulator >= tick) {
      this.player.step(control, tick, this.world);
      this.accumulator -= tick;
    }

    this.handleInteractions(control);
    this.world.update(this.player.position);
    const instances = this.world.gatherInstances();
    this.renderer.render(instances.translations, instances.blocks, instances.lights, instances.count, this.player.position, this.input.look[0], this.input.look[1]);
    const fps = 1 / Math.max(dt, 1e-3);
    this.ui.draw(fps, this.world.chunks.size, this.selectedBlock.name, this.player.flying);
    requestAnimationFrame((t) => this.loop(t));
  }

  /**
   * Handle mining/placing interactions from input state.
   * @param {ReturnType<InputManager['sample']>} control
   */
  handleInteractions(control) {
    const eye = new Float32Array([
      this.player.position[0],
      this.player.position[1] + PLAYER_EYE,
      this.player.position[2],
    ]);
    const dir = this.player.forward();
    const hit = raycast(this.world, eye, dir, INTERACT_RANGE);
    if (!hit.hit) return;
    if (control.breakBlock) {
      this.world.setBlock(hit.position[0], hit.position[1], hit.position[2], 0);
    } else if (control.placeBlock) {
      const targetPos = [hit.position[0] + hit.normal[0], hit.position[1] + hit.normal[1], hit.position[2] + hit.normal[2]];
      if (targetPos[1] >= 0 && targetPos[1] < 255) {
        this.world.setBlock(targetPos[0], targetPos[1], targetPos[2], this.selectedBlock.id);
      }
    }
  }
}

window.addEventListener('DOMContentLoaded', () => {
  // eslint-disable-next-line no-new
  new App();
});
