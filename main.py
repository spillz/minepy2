import math
import random
import time
import sys

# pyglet imports
import pyglet
image = pyglet.image
from pyglet.window import key, mouse
import pyglet.gl as gl
import pyglet.shapes as shapes
from pyglet.math import Mat4, Vec3

GLfloat3 = gl.GLfloat*3
GLfloat4 = gl.GLfloat*4

# standard lib imports
from collections import deque
import numpy
import itertools

# local module imports
import world_proxy as world
import util
import config
import shaders
from blocks import TEXTURE_PATH
from config import DIST, TICKS_PER_SEC, FLYING_SPEED, GRAVITY, JUMP_SPEED, \
        MAX_JUMP_HEIGHT, PLAYER_HEIGHT, TERMINAL_VELOCITY, TICKS_PER_SEC, \
        WALKING_SPEED
from blocks import BLOCK_ID, BLOCK_TEXTURES, BLOCK_VERTICES, BLOCK_COLORS, BLOCK_SOLID, BLOCK_PICKER_FACE
WATER = BLOCK_ID['Water']


class Window(pyglet.window.Window):

    def __init__(self, *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)


        # Whether or not the window exclusively captures the mouse.
        self.exclusive = False

        # When flying gravity has no effect and speed is increased.
        self.flying = True
        self.fly_climb = 0

        # Strafing is moving lateral to the direction you are facing,
        # e.g. moving to the left or right while continuing to face forward.
        #
        # First element is -1 when moving forward, 1 when moving back, and 0
        # otherwise. The second element is -1 when moving left, 1 when moving
        # right, and 0 otherwise.
        self.strafe = [0, 0]

        # Current (x, y, z) position in the world, specified with floats. Note
        # that, perhaps unlike in math class, the y-axis is the vertical axis.
        if config.DEBUG_SINGLE_BLOCK:
            bx = config.SECTOR_SIZE//2
            by = config.SECTOR_HEIGHT//2
            bz = config.SECTOR_SIZE//2 + 2
            self.position = (bx, by, bz)
            print(f"[DEBUG] Starting at {self.position} for single-block debug")
        else:
            self.position = (0, 160, 0)

        # First element is rotation of the player in the x-z plane (ground
        # plane) measured from the z-axis down. The second is the rotation
        # angle from the ground plane up. Rotation is in degrees.
        #
        # The vertical plane rotation ranges from -90 (looking straight down) to
        # 90 (looking straight up). The horizontal rotation range is unbounded.
        self.rotation = (0, 0)

        # Which sector the player is currently in.
        self.sector = None

        # The crosshairs at the center of the screen.
        self.reticle = []

        self.inventory_item = None

        # Debug flags
        self._printed_mats = False

        # Velocity in the y (upward) direction.
        self.dy = 0

        # A list of blocks the player can place. Hit num keys to cycle.
        self.inventory = list(BLOCK_ID)

        # The current block the user can place. Hit num keys to cycle.
        self.block = self.inventory[0]

        # Convenience list of num keys.
        self.num_keys = [
            key._1, key._2, key._3, key._4, key._5,
            key._6, key._7, key._8, key._9, key._0]

        # Shader program used for world rendering.
        self.block_program = shaders.create_block_shader()
        self.block_program['u_texture'] = 0
        self.block_program['u_light_dir'] = (0.35, 1.0, 0.65)
        self.block_program['u_fog_color'] = (0.5, 0.69, 1.0)
        if config.DEBUG_SINGLE_BLOCK:
            self.block_program['u_fog_start'] = 1e6
            self.block_program['u_fog_end'] = 2e6
        else:
            self.block_program['u_fog_start'] = 0.75 * DIST
            self.block_program['u_fog_end'] = DIST

        # Instance of the model that handles the world.
        self.model = world.ModelProxy(self.block_program)

        # Texture atlas for UI previews.
        self.texture_atlas = image.load(TEXTURE_PATH)

        # The label that is displayed in the top left of the canvas.
        self.label = pyglet.text.Label('', font_name='Arial', font_size=18,
            x=10, y=self.height - 10, anchor_x='left', anchor_y='top',
            color=(0, 0, 0, 255))

        # Target frame pacing and local FPS tracking (not dependent on pyglet internals).
        desired_fps = getattr(config, 'TARGET_FPS', None)
        self.target_fps = desired_fps or self._detect_refresh_rate() or 60
        self._frame_times = deque(maxlen=120)
        self._last_frame_time = time.perf_counter()
        # Use pyglet's clock-based limiter; avoid double-limiting with manual sleeps.
        try:
            pyglet.clock.set_fps_limit(self.target_fps)
        except Exception:
            pass

        # This call schedules the `update()` method to be called
        # TICKS_PER_SEC. This is the main game event loop.
        pyglet.clock.schedule_interval(self.update, 1.0 / TICKS_PER_SEC)

    def set_exclusive_mouse(self, exclusive):
        """ If `exclusive` is True, the game will capture the mouse, if False
        the game will ignore the mouse.

        """
        super(Window, self).set_exclusive_mouse(exclusive)
        self.exclusive = exclusive

    def get_sight_vector(self):
        """ Returns the current line of sight vector indicating the direction
        the player is looking.

        """
        x, y = self.rotation
        # y ranges from -90 to 90, or -pi/2 to pi/2, so m ranges from 0 to 1 and
        # is 1 when looking ahead parallel to the ground and 0 when looking
        # straight up or down.
        m = math.cos(math.radians(y))
        # dy ranges from -1 to 1 and is -1 when looking straight down and 1 when
        # looking straight up.
        dy = math.sin(math.radians(y))
        dx = math.cos(math.radians(x - 90)) * m
        dz = math.sin(math.radians(x - 90)) * m
        return (dx, dy, dz)

    def get_motion_vector(self):
        """ Returns the current motion vector indicating the velocity of the
        player.

        Returns
        -------
        vector : tuple of len 3
            Tuple containing the velocity in x, y, and z respectively.

        """
        if any(self.strafe):
            x, y = self.rotation
            strafe = math.degrees(math.atan2(*self.strafe))
            y_angle = math.radians(y)*any(self.strafe)
            x_angle = math.radians(x + strafe)*any(self.strafe)
            if self.flying:
                m = math.cos(y_angle)
                dy = math.sin(y_angle)
                if self.strafe[1]:
                    # Moving left or right.
                    dy = 0.0
                    m = 1
                if self.strafe[0] > 0:
                    # Moving backwards.
                    dy *= -1
                # When you are flying up or down, you have less left and right
                # motion.
                dx = math.cos(x_angle) * m
                dz = math.sin(x_angle) * m
            else:
                dy = 0.0
                dx = math.cos(x_angle)
                dz = math.sin(x_angle)
        else:
            dy = 0.0
            dx = 0.0
            dz = 0.0
        if self.flying and self.fly_climb!=0:
            dy = self.fly_climb
        return (dx, dy, dz)

    def update(self, dt):
        """ This method is scheduled to be called repeatedly by the pyglet
        clock.

        Parameters
        ----------
        dt : float
            The change in time since the last call.

        """
#        self.model.process_queue()
        sector = util.sectorize(self.position)
        if self.model.loader is not None:
            look_vec = self.get_sight_vector()
            self.model.update_sectors(self.sector, sector, self.position, look_vec)
            self.sector = sector
        m = 20
        dt = min(dt, 0.2)
        for _ in range(m):
            self._update(dt / m)

    def _update(self, dt):
        """ Private implementation of the `update()` method. This is where most
        of the motion logic lives, along with gravity and collision detection.

        Parameters
        ----------
        dt : float
            The change in time since the last call.

        """
        # walking
        speed = FLYING_SPEED if self.flying else WALKING_SPEED
        d = dt * speed # distance covered this tick.
        dx, dy, dz = self.get_motion_vector()
        # New position in space, before accounting for gravity.
        dx, dy, dz = dx * d, dy * d, dz * d
        # gravity
        if not self.flying:
            # Update your vertical speed: if you are falling, speed up until you
            # hit terminal velocity; if you are jumping, slow down until you
            # start falling.
            self.dy -= dt * GRAVITY
            self.dy = max(self.dy, -TERMINAL_VELOCITY)
            dy += self.dy * dt
        # collisions
        x, y, z = self.position
        x, y, z = self.collide((x + dx, y + dy, z + dz), PLAYER_HEIGHT)
        self.position = (x, y, z)

    def collide(self, position, height):
        """ Checks to see if the player at the given `position` and `height`
        is colliding with any blocks in the world.

        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position to check for collisions at.
        height : int or float
            The height of the player.

        Returns
        -------
        position : tuple of len 3
            The new position of the player taking into account collisions.

        """
        # How much overlap with a dimension of a surrounding block you need to
        # have to count as a collision. If 0, touching terrain at all counts as
        # a collision. If .49, you sink into the ground, as if walking through
        # tall grass. If >= .5, you'll fall through the ground.
        pad = 0.1
        p = list(position)
        np = util.normalize(position)
        for face in util.FACES:  # check all surrounding blocks
            for i in range(3):  # check each dimension independently
                if not face[i]:
                    continue
                # How much overlap you have with this dimension.
                d = (p[i] - np[i]) * face[i]
                if d < pad:
                    continue
                for dy in range(height):  # check each height
                    op = list(np)
                    op[1] -= dy
                    op[i] += face[i]
                    b = self.model[util.normalize(op)]
                    if b is None or b==0 or not BLOCK_SOLID[b]:
                        continue
                    p[i] -= (d - pad) * face[i]
                    if face == (0, -1, 0) or face == (0, 1, 0):
                        # You are colliding with the ground or ceiling, so stop
                        # falling / rising.
                        self.dy = 0
                    break
        return tuple(p)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        ind = self.inventory.index(self.block)
        step = int(scroll_y)
        if step == 0:
            step = 1 if scroll_y > 0 else -1
        ind += step
        if ind >= len(self.inventory):
            ind -= len(self.inventory)
        if ind < 0:
            ind += len(self.inventory)
        self.block = self.inventory[ind]
        self.update_inventory_item_batch()

    def on_mouse_press(self, x, y, button, modifiers):
        """ Called when a mouse button is pressed. See pyglet docs for button
        amd modifier mappings.

        Parameters
        ----------
        x, y : int
            The coordinates of the mouse click. Always center of the screen if
            the mouse is captured.
        button : int
            Number representing mouse button that was clicked. 1 = left button,
            4 = right button.
        modifiers : int
            Number representing any modifying keys that were pressed when the
            mouse button was clicked.

        """
        if self.exclusive:
            vector = self.get_sight_vector()
            block, previous = self.model.hit_test(self.position, vector)
            if (button == mouse.RIGHT) or \
                    ((button == mouse.LEFT) and (modifiers & key.MOD_CTRL)):
                # ON OSX, control + left click = right click.
                if previous:
                    px, py, pz = util.normalize(self.position)
                    if not (previous == (px, py, pz) or previous == (px, py-1, pz)):
                        self.model.add_block(previous, BLOCK_ID[self.block])
            elif button == pyglet.window.mouse.LEFT and block:
                self.model.remove_block(block)
        else:
            self.set_exclusive_mouse(True)

    def on_mouse_motion(self, x, y, dx, dy):
        """ Called when the player moves the mouse.

        Parameters
        ----------
        x, y : int
            The coordinates of the mouse click. Always center of the screen if
            the mouse is captured.
        dx, dy : float
            The movement of the mouse.

        """
        if self.exclusive:
            m = 0.15
            x, y = self.rotation  # x = yaw (degrees), y = pitch (degrees)
            x = x + dx * m
            y = max(-90, min(90, y + dy * m))
            self.rotation = (x, y)
            if config.DEBUG_SINGLE_BLOCK:
                print(f"[DEBUG] rotation yaw={x:.2f} pitch={y:.2f}")

    def on_key_press(self, symbol, modifiers):
        """ Called when the player presses a key. See pyglet docs for key
        mappings.

        Parameters
        ----------
        symbol : int
            Number representing the key that was pressed.
        modifiers : int
            Number representing any modifying keys that were pressed.

        """
        if symbol == key.W:
            self.strafe[0] -= 1
        elif symbol == key.S:
            self.strafe[0] += 1
        elif symbol == key.A:
            self.strafe[1] -= 1
        elif symbol == key.D:
            self.strafe[1] += 1
        elif symbol == key.SPACE:
            if self.flying:
                self.fly_climb += 1
            if self.dy == 0:
                self.dy = JUMP_SPEED
        elif symbol == key.LSHIFT:
            if self.flying:
                self.fly_climb -= 1
        elif symbol == key.ESCAPE:
            self.set_exclusive_mouse(False)
        elif symbol == key.TAB:
            self.flying = not self.flying
        elif symbol in self.num_keys:
            index = (symbol - self.num_keys[0]) % len(self.inventory)
            self.block = self.inventory[index]
            self.update_inventory_item_batch()

    def on_key_release(self, symbol, modifiers):
        """ Called when the player releases a key. See pyglet docs for key
        mappings.

        Parameters
        ----------
        symbol : int
            Number representing the key that was pressed.
        modifiers : int
            Number representing any modifying keys that were pressed.

        """
        if symbol == key.W:
            self.strafe[0] += 1
        elif symbol == key.S:
            self.strafe[0] -= 1
        elif symbol == key.A:
            self.strafe[1] += 1
        elif symbol == key.D:
            self.strafe[1] -= 1
        elif symbol == key.SPACE:
            self.fly_climb = 0
        elif symbol == key.LSHIFT:
            self.fly_climb = 0

    def on_resize(self, width, height):
        """ Called when the window is resized to a new `width` and `height`.

        """
        # label
        self.label.y = height - 10
        # reticle uses shader-based shapes instead of deprecated vertex_list
        self.reticle_batch = None
        cx, cy = self.width / 2, self.height / 2
        n = 10
        self.reticle = [
            shapes.Line(cx - n, cy, cx + n, cy, thickness=2, color=(0, 0, 0)),
            shapes.Line(cx, cy - n, cx, cy + n, thickness=2, color=(0, 0, 0)),
        ]
        #inventory item
        self.update_inventory_item_batch()

    def update_inventory_item_batch(self):
        if self.inventory_item is not None:
            self.inventory_item.delete()
        # Draw a textured preview of the selected block using the atlas.
        block_id = BLOCK_ID[self.block]
        picker_face = int(BLOCK_PICKER_FACE[block_id])
        t = BLOCK_TEXTURES[block_id][picker_face]  # configured face tex coords
        tex = self.texture_atlas.get_texture()
        x0, y0, x1, y1 = t[0]*tex.width, t[1]*tex.height, t[4]*tex.width, t[5]*tex.height
        region = tex.get_region(x=int(x0), y=int(y0), width=int(x1 - x0), height=int(y1 - y0))
        size = 64
        sprite = pyglet.sprite.Sprite(region, x=16, y=16)
        scale_x = size / sprite.width
        scale_y = size / sprite.height
        sprite.scale = min(scale_x, scale_y)
        self.inventory_item = sprite
        #outline
#        v = size/2+(size/2+0.1)*BLOCK_VERTICES[BLOCK_ID[self.block]] + numpy.tile(numpy.array([16,16+size/2,0]),4)
#        v = numpy.hstack((v[:,:3],v,v[:,-3:]))
#        v = v.ravel()
#        c = 1*numpy.array([1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1]).repeat(6)
#        self.inventory_item_outline = self.inventory_batch.add(len(v)/3, gl.GL_LINE_STRIP, self.inventory_outline_group,
#            ('v3f/static', v),
#            ('c3B/static', c),
#        )

    def on_close(self):
        self.model.quit()
        pyglet.window.Window.on_close(self)

    def set_2d(self):
        """ Configure OpenGL to draw in 2d.

        """
        width, height = self.get_size()
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glViewport(0, 0, width, height)

    def get_view_projection(self):
        """Return projection and view matrices using pyglet Mat4 (column-major)."""
        width, height = self.get_size()
        aspect = width / float(height)
        near, far = 0.1, 512.0
        projection = Mat4.perspective_projection(aspect, near, far, 65)
        x, y, z = self.position
        dx, dy, dz = self.get_sight_vector()
        eye = Vec3(x, y, z)
        forward = Vec3(dx, dy, dz)
        # Avoid the look_at up vector becoming collinear with the view direction
        # when the player looks straight up/down, which would produce a bad view matrix.
        up = Vec3(0.0, 1.0, 0.0) if abs(dy) < 0.99 else Vec3(0.0, 0.0, 1.0)
        target = eye + forward
        view = Mat4.look_at(eye, target, up)
        return projection, view

    def set_3d(self):
        """ Configure OpenGL to draw in 3d.

        """
        width, height = self.get_size()

        gl.glEnable(gl.GL_DEPTH_TEST)

        gl.glViewport(0, 0, width, height)
        projection, view = self.get_view_projection()
        # pyglet Mat4 supports direct upload; ensure contiguous float32 arrays
        self.model.set_matrices(projection, view)
        if config.DEBUG_SINGLE_BLOCK and not self._printed_mats:
            dx, dy, dz = self.get_sight_vector()
            print("[DEBUG] projection matrix:\n", numpy.array(projection))
            print("[DEBUG] view matrix:\n", numpy.array(view))
            print(f"[DEBUG] position {self.position} rotation {self.rotation} sight {(dx, dy, dz)}")
            self._printed_mats = True
##        gl.glLightfv(gl.GL_LIGHT1, gl.GL_POSITION, GLfloat4(0.35,1.0,0.65,0.0))
        #gl.glLightfv(gl.GL_LIGHT0,gl.GL_SPECULAR, GLfloat4(1,1,1,1))


    def get_frustum_circle(self):
        x,y = self.rotation
        dx = math.cos(math.radians(x - 90))
        dz = math.sin(math.radians(x - 90))

        c = [0,2]
        vec = numpy.array([dx,dz])
        ovec = numpy.array([-dz,dx])
        pos = numpy.array([x for x in self.position])[c]
        center = pos + vec*DIST/2
        far_corner = pos + vec*DIST + ovec*DIST*numpy.tan(65.0/180.0 * numpy.pi)/2
        rad = ((center-far_corner)**2).sum()**0.5/2
        return center,rad

    def on_draw(self):
        """ Called by pyglet to draw the canvas.

        """
        frame_start = time.perf_counter()
        dt = frame_start - self._last_frame_time
        self._last_frame_time = frame_start
        self._frame_times.append(dt)
        # Allow a small slice of the frame for mesh uploads; keep rendering priority.
        frame_budget = 1.0 / self.target_fps if self.target_fps else 1.0 / 60.0
        # upload_budget = 0.3 * frame_budget
        upload_budget = max(0.5/self.target_fps, 0.5 * frame_budget)
        self.clear()
        self.set_3d()
        # Defer mesh uploads until after rendering so we know exactly how much time remains.
        self.model.draw(self.position, self.get_frustum_circle(), frame_start, upload_budget, defer_uploads=True)
        self.set_2d()
        self.draw_label()
        self.draw_reticle()
        self.draw_inventory_item()
        self.draw_focused_block()
        if self._is_underwater():
            self.draw_underwater_overlay()
        # Use leftover budget to upload meshes at the end of the frame.
        elapsed = time.perf_counter() - frame_start
        remaining_upload_budget = max(0.0, upload_budget - elapsed)
        if remaining_upload_budget > 0:
            self.model.process_pending_uploads(frame_start, remaining_upload_budget)

    def draw_label(self):
        """ Draw the label in the top left of the screen.

        """
        x, y, z = self.position
        rx, ry = self.rotation
        fps = self._current_fps()
        # Void probe: count solid blocks along reticle until next air.
        sight = self.get_sight_vector()
        void_dist = self.model.measure_void_distance(self.position, sight, max_distance=64)
        if void_dist is None:
            void_text = 'N/A'
        elif void_dist >= 64:
            void_text = '>=64'
        else:
            void_text = str(void_dist)
        self.label.text = 'FPS: %.1f  (%.2f, %.2f, %.2f) rot(%.1f, %.1f) void %s' % (fps, x, y, z, rx, ry, void_text)
        # Light backdrop to keep text readable on bright backgrounds.
        pad_x = 6
        pad_y = 3
        bg_width = self.label.content_width + pad_x * 2
        bg_height = self.label.content_height + pad_y * 2
        bg_x = self.label.x - pad_x
        bg_y = self.label.y - bg_height
        label_bg = shapes.Rectangle(bg_x, bg_y, bg_width, bg_height, color=(255, 255, 255))
        label_bg.opacity = 120  # semi-transparent
        label_bg.draw()
        self.label.draw()

    def _current_fps(self):
        """Return a smoothed FPS based on recent draw intervals."""
        if not self._frame_times:
            return 0.0
        total = sum(self._frame_times)
        return len(self._frame_times) / total if total > 0 else 0.0

    def _detect_refresh_rate(self):
        """Try to read the current monitor refresh rate for frame capping."""
        try:
            screen = self.display.get_default_screen()
            mode = screen.get_mode()
            rate = getattr(mode, 'refresh_rate', None)
            if rate:
                return rate
        except Exception:
            print('Refresh rate not detected')
            pass
        return None

    def draw_focused_block(self):
        """ Draw edges around the block under the crosshairs for placement feedback. """
        return  # temporarily disabled due to pyglet draw API mismatch

    def draw_reticle(self):
        """ Draw the crosshairs in the center of the screen.

        """
        for line in self.reticle:
            line.draw()

    def draw_inventory_item(self):
        if self.inventory_item:
            self.inventory_item.draw()

    def _is_underwater(self):
        """Return True if the camera is currently inside a water block."""
        pos = util.normalize(self.position)
        if self.model[pos] == WATER:
            return True
        head_pos = (pos[0], pos[1] + 1, pos[2])
        return self.model[head_pos] == WATER

    def draw_underwater_overlay(self):
        """Render a full-viewport tint when submerged to avoid per-block transparency."""
        width, height = self.get_size()
        overlay = shapes.Rectangle(0, 0, width, height, color=(40, 110, 170))
        overlay.opacity = 90
        overlay.draw()



def setup_fog():
    """ Configure the OpenGL fog properties.

    """
    # Fixed-function fog isn't available on modern/core profiles; skip if missing.
    return



def setup():
    """ Basic OpenGL configuration.

    """
    # Set the color of "clear", i.e. the sky, in rgba.
    gl.glClearColor(0.5, 0.69, 1.0, 1)
    # Cull back faces for better fill-rate; geometry is built with consistent winding.
    gl.glEnable(gl.GL_CULL_FACE)
    gl.glCullFace(gl.GL_BACK)

    #gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_DST_ALPHA)
    #gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
#    gl.glBlendFunc(gl.GL_ZERO, gl.GL_SRC_COLOR)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    # Set the texture minification/magnification function to GL_NEAREST (nearest
    # in Manhattan distance) to the specified texture coordinates. GL_NEAREST
    # "is generally faster than GL_LINEAR, but it can produce textured images
    # with sharper edges because the transition between texture elements is not
    # as smooth."
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    # Fixed-function texture env not needed with shaders.
    setup_fog()


def main():
    if len(sys.argv)>1:
        arg = sys.argv[1]
        if ':' in arg:
            host, port = arg.split(':', 1)
            config.SERVER_IP = host
            try:
                config.SERVER_PORT = int(port)
            except ValueError:
                pass
        else:
            config.SERVER_IP = arg
        print('Using server IP address',config.SERVER_IP,':',config.SERVER_PORT)
    window = Window(width=300, height=200, caption='Pyglet', resizable=True, vsync=True)
    # Hide the mouse cursor and prevent the mouse from leaving the window.
    window.set_exclusive_mouse(True)
    setup()
    try:
        pyglet.app.run()
    except:
        import traceback
        traceback.print_exc()
        print('terminating child processes')
        window.model.quit()
        window.set_exclusive_mouse(False)


if __name__ == '__main__':
    main()
