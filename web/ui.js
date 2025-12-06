/**
 * Simple 2D canvas overlay for UI elements.
 * @module ui
 */
import { UI_FONT } from './config.js';

export class UIOverlay {
  /**
   * @param {HTMLCanvasElement} canvas - 2D canvas element.
   */
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.lastFps = 0;
  }

  /**
   * Resize the backing store to match the WebGL canvas.
   * @param {number} width - Width in pixels.
   * @param {number} height - Height in pixels.
   */
  resize(width, height) {
    this.canvas.width = width;
    this.canvas.height = height;
  }

  /**
   * Render overlay information.
   * @param {number} fps - Frames per second.
   * @param {number} chunkCount - Loaded chunk count.
   * @param {string} selectedBlock - Current hotbar selection name.
   * @param {boolean} flying - Whether the player is flying.
   */
  draw(fps, chunkCount, selectedBlock, flying) {
    const { ctx } = this;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 2;
    const cx = this.canvas.width / 2;
    const cy = this.canvas.height / 2;
    ctx.beginPath();
    ctx.moveTo(cx - 10, cy);
    ctx.lineTo(cx + 10, cy);
    ctx.moveTo(cx, cy - 10);
    ctx.lineTo(cx, cy + 10);
    ctx.stroke();

    ctx.fillStyle = '#000';
    ctx.font = UI_FONT;
    ctx.fillText(`FPS: ${fps.toFixed(1)}`, 10, 20);
    ctx.fillText(`Chunks: ${chunkCount}`, 10, 40);
    ctx.fillText(`Block: ${selectedBlock}`, 10, 60);
    ctx.fillText(`Mode: ${flying ? 'Flying' : 'Walking'}`, 10, 80);
  }
}
