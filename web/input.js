/**
 * Keyboard and pointer capture helpers.
 * @module input
 */
import { clamp } from './util.js';

/**
 * Handles keyboard and pointer events.
 */
export class InputManager {
  /**
   * @param {HTMLElement} element - Element to capture events from.
   */
  constructor(element) {
    this.element = element;
    this.keys = new Set();
    this.look = new Float32Array([0, 0]);
    this.pointerLocked = false;
    this.jumpRequested = false;
    this.primaryDown = false;
    this.secondaryDown = false;
    this.toggleFly = false;
    this.hotbarIndex = 0;
    this._initListeners();
  }

  _initListeners() {
    window.addEventListener('keydown', (e) => {
      this.keys.add(e.code);
      if (e.code === 'KeyF') {
        this.toggleFly = true;
      }
    });
    window.addEventListener('keyup', (e) => {
      this.keys.delete(e.code);
      if (e.code === 'Space') {
        this.jumpRequested = false;
      }
    });
    this.element.addEventListener('mousedown', (e) => {
      if (e.button === 0) this.primaryDown = true;
      if (e.button === 2) this.secondaryDown = true;
    });
    this.element.addEventListener('mouseup', (e) => {
      if (e.button === 0) this.primaryDown = false;
      if (e.button === 2) this.secondaryDown = false;
    });
    this.element.addEventListener('contextmenu', (e) => e.preventDefault());
    this.element.addEventListener('wheel', (e) => {
      if (e.deltaY > 0) {
        this.hotbarIndex = (this.hotbarIndex + 1) % 9;
      } else if (e.deltaY < 0) {
        this.hotbarIndex = (this.hotbarIndex + 8) % 9;
      }
    });
    this.element.addEventListener('click', () => {
      if (!this.pointerLocked) {
        this.element.requestPointerLock();
      }
    });
    document.addEventListener('pointerlockchange', () => {
      this.pointerLocked = document.pointerLockElement === this.element;
    });
    window.addEventListener('mousemove', (e) => {
      if (!this.pointerLocked) return;
      this.look[0] += e.movementX * 0.0025;
      this.look[1] = clamp(this.look[1] - e.movementY * 0.0025, -Math.PI / 2, Math.PI / 2);
    });
  }

  /**
   * Compute the current directional intent.
   * @returns {{forward: number, right: number, jump: boolean, sprint: boolean, crouch: boolean, breakBlock: boolean, placeBlock: boolean, toggleFly: boolean, hotbar: number}}
   */
  sample() {
    const forward = (this.keys.has('KeyW') ? 1 : 0) + (this.keys.has('KeyS') ? -1 : 0);
    const right = (this.keys.has('KeyD') ? 1 : 0) + (this.keys.has('KeyA') ? -1 : 0);
    const jump = this.keys.has('Space');
    const sprint = this.keys.has('ShiftLeft') || this.keys.has('ShiftRight');
    const crouch = this.keys.has('ControlLeft') || this.keys.has('ControlRight');
    const toggleFly = this.toggleFly;
    this.toggleFly = false;
    return {
      forward,
      right,
      jump,
      sprint,
      crouch,
      breakBlock: this.primaryDown,
      placeBlock: this.secondaryDown,
      toggleFly,
      hotbar: this.hotbarIndex,
    };
  }
}
