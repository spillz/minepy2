/**
 * Player physics state and camera orientation helpers.
 * @module player
 */
import { GRAVITY, JUMP_SPEED, WALK_SPEED, FLY_SPEED, PLAYER_WIDTH, PLAYER_HEIGHT } from './config.js';
import { BLOCK_SOLID } from './blocks.js';
import { clamp } from './util.js';

/**
 * Player physics state and camera orientation.
 */
export class Player {
  constructor() {
    this.position = new Float32Array([0, 40, 0]);
    this.velocity = new Float32Array([0, 0, 0]);
    this.yaw = 0;
    this.pitch = 0;
    this.flying = false;
    this.onGround = false;
  }

  /**
   * Integrate motion for a tick.
   * @param {{forward: number, right: number, jump: boolean, sprint: boolean, crouch: boolean, toggleFly: boolean}} input - Movement intent.
   * @param {number} dt - Delta time in seconds.
   * @param {import('./world.js').World} world - World accessor for collisions.
   */
  step(input, dt, world) {
    if (input.toggleFly) {
      this.flying = !this.flying;
      this.velocity[1] = 0;
    }
    this.yaw = input.lookYaw ?? this.yaw;
    this.pitch = input.lookPitch ?? this.pitch;
    const speed = this.flying ? FLY_SPEED : (input.sprint ? WALK_SPEED * 1.6 : WALK_SPEED);
    const dirX = Math.cos(this.yaw);
    const dirZ = Math.sin(this.yaw);
    const forwardX = dirX * input.forward;
    const forwardZ = dirZ * input.forward;
    const rightX = dirZ * input.right;
    const rightZ = -dirX * input.right;
    let vx = forwardX + rightX;
    let vz = forwardZ + rightZ;
    const len = Math.hypot(vx, vz);
    if (len > 0) {
      vx = (vx / len) * speed;
      vz = (vz / len) * speed;
    }

    this.velocity[0] = vx;
    this.velocity[2] = vz;

    if (this.flying) {
      this.velocity[1] = (input.jump ? speed : 0) + (input.crouch ? -speed : 0);
    } else {
      this.velocity[1] -= GRAVITY * dt;
      if (input.jump && this.onGround) {
        this.velocity[1] = JUMP_SPEED;
      }
    }

    this.move(world, this.velocity[0] * dt, 0, 0);
    this.move(world, 0, this.velocity[1] * dt, 0);
    this.move(world, 0, 0, this.velocity[2] * dt);

    this.pitch = clamp(this.pitch, -Math.PI / 2, Math.PI / 2);
  }

  /**
   * Move along one axis while resolving collisions.
   * @param {import('./world.js').World} world
   * @param {number} dx
   * @param {number} dy
   * @param {number} dz
   */
  move(world, dx, dy, dz) {
    const next = new Float32Array([this.position[0] + dx, this.position[1] + dy, this.position[2] + dz]);
    const halfW = PLAYER_WIDTH / 2;
    const minX = Math.floor(next[0] - halfW);
    const maxX = Math.floor(next[0] + halfW);
    const minY = Math.floor(next[1]);
    const maxY = Math.floor(next[1] + PLAYER_HEIGHT);
    const minZ = Math.floor(next[2] - halfW);
    const maxZ = Math.floor(next[2] + halfW);
    let collided = false;
    for (let y = minY; y <= maxY; y++) {
      for (let z = minZ; z <= maxZ; z++) {
        for (let x = minX; x <= maxX; x++) {
          const b = world.getBlock(x, y, z);
          if (b !== 0 && BLOCK_SOLID[b]) {
            collided = true;
            if (dx > 0) next[0] = x - halfW - 1e-3;
            if (dx < 0) next[0] = x + 1 + halfW + 1e-3;
            if (dz > 0) next[2] = z - halfW - 1e-3;
            if (dz < 0) next[2] = z + 1 + halfW + 1e-3;
            if (dy > 0) {
              next[1] = y - PLAYER_HEIGHT - 1e-3;
              this.velocity[1] = Math.min(0, this.velocity[1]);
            }
            if (dy < 0) {
              next[1] = y + 1 + 1e-3;
              this.velocity[1] = Math.max(0, this.velocity[1]);
              this.onGround = true;
            }
          }
        }
      }
    }
    if (!collided && dy < 0) {
      this.onGround = false;
    }
    this.position.set(next);
  }

  /**
   * Forward direction vector based on orientation.
   * @returns {Float32Array}
   */
  forward() {
    const cp = Math.cos(this.pitch);
    return new Float32Array([
      Math.cos(this.yaw) * cp,
      Math.sin(this.pitch),
      Math.sin(this.yaw) * cp,
    ]);
  }
}
