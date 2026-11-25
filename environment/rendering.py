"""Renderers for Medicine Drone (retro top-down look, no image files needed).

DroneRenderer draws the scene with OpenGL (PyOpenGL), in a pygame-created GL window.
If no OpenGL context can be created (for example on a server without a display), it falls back to
_SoftwareRenderer, the same picture drawn with plain pygame, so training and tests still work."""
import math

import numpy as np
import pygame

try:
    from OpenGL import GL as gl
except Exception:                                   # PyOpenGL missing: software fallback only
    gl = None

CELL, TOP = 64, 70


class _SoftwareRenderer:
    def __init__(self, mode, fps):
        pygame.init()
        self.mode, self.fps = mode, fps
        self.window = None
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 15, bold=True)
        self.small = pygame.font.SysFont("monospace", 12, bold=True)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
        pygame.quit()

    def _bar(self, s, x, y, w, frac, col, h=7):
        pygame.draw.rect(s, (40, 40, 40), (x, y, w, h))
        pygame.draw.rect(s, col, (x, y, int(w * max(0.0, min(1.0, frac))), h))

    def draw(self, env):
        n = env.size
        w, h = n * CELL, n * CELL + TOP
        surf = pygame.Surface((w, h))
        surf.fill((30, 34, 46))
        for x in range(n):                      # green countryside
            for y in range(n):
                c = (112, 168, 92) if (x + y) % 2 == 0 else (104, 160, 86)
                pygame.draw.rect(surf, c, (x * CELL, TOP + y * CELL, CELL, CELL))
        for x in range(n):                      # hills
            for y in range(n):
                if env.hills[x, y]:
                    cx, cy = x * CELL + CELL // 2, TOP + y * CELL + CELL // 2
                    pygame.draw.polygon(surf, (128, 100, 70), [(cx - 26, cy + 18), (cx, cy - 20), (cx + 26, cy + 18)])
                    pygame.draw.polygon(surf, (158, 128, 92), [(cx - 10, cy + 2), (cx, cy - 20), (cx + 6, cy - 6)])
        # hub: a hospital depot with a red cross
        hx, hy = env.hub
        bx, by = hx * CELL + 4, TOP + hy * CELL + 6
        pygame.draw.rect(surf, (240, 240, 240), (bx, by, CELL - 8, CELL - 12), border_radius=4)
        pygame.draw.rect(surf, (210, 40, 40), (bx + 22, by + 8, 14, 34)); pygame.draw.rect(surf, (210, 40, 40), (bx + 12, by + 18, 34, 14))
        t = self.small.render("HUB", True, (20, 20, 20)); surf.blit(t, (bx + 17, by + CELL - 24))
        # clinics with stock bars
        for k in range(env.nc):
            x, y = env.cx[k], env.cy[k]
            px, py = x * CELL, TOP + y * CELL
            low = env.stock[k] < 0.2
            pygame.draw.rect(surf, (200, 70, 60) if low else (230, 215, 180), (px + 10, py + 22, CELL - 20, CELL - 30), border_radius=3)
            pygame.draw.polygon(surf, (150, 60, 50), [(px + 6, py + 24), (px + CELL // 2, py + 6), (px + CELL - 6, py + 24)])
            pygame.draw.rect(surf, (60, 40, 30), (px + CELL // 2 - 6, py + 38, 12, 16))
            self._bar(surf, px + 8, py + 2, CELL - 16, env.stock[k], (230, 60, 50) if low else (80, 200, 110))
        # storms
        for sx, sy in env.storms:
            cx, cy = sx * CELL + CELL // 2, TOP + sy * CELL + CELL // 2
            for ox, oy, r in ((-14, 4, 16), (6, -2, 20), (18, 6, 14), (0, 10, 16)):
                pygame.draw.circle(surf, (70, 74, 96), (cx + ox, cy + oy), r)
            pygame.draw.polygon(surf, (250, 220, 60), [(cx, cy + 6), (cx - 7, cy + 22), (cx - 1, cy + 22), (cx - 5, cy + 34), (cx + 8, cy + 16), (cx + 2, cy + 16)])
        # drone
        gx, gy = env.dx * CELL + CELL // 2, TOP + env.dy * CELL + CELL // 2
        pygame.draw.ellipse(surf, (80, 120, 70), (gx - 14, gy + 18, 28, 8))
        for ox, oy in ((-16, -14), (16, -14), (-16, 12), (16, 12)):
            pygame.draw.line(surf, (50, 50, 60), (gx, gy), (gx + ox, gy + oy), 3)
            pygame.draw.circle(surf, (190, 190, 205), (gx + ox, gy + oy), 8)
            pygame.draw.circle(surf, (60, 60, 70), (gx + ox, gy + oy), 8, 2)
        pygame.draw.rect(surf, (35, 90, 200), (gx - 9, gy - 9, 18, 18), border_radius=4)
        for b in range(env.load):
            pygame.draw.rect(surf, (245, 245, 245), (gx - 8 + b * 9, gy + 14, 8, 8)); pygame.draw.rect(surf, (210, 40, 40), (gx - 5 + b * 9, gy + 15, 2, 6))
        self._bar(surf, gx - 18, gy - 26, 36, env.battery / env.max_battery, (250, 200, 50) if env.battery > 12 else (230, 60, 50))
        # status bar
        s = env.stats
        surf.blit(self.font.render(f"step {env.t}/{env.day_length}   battery {env.battery:4.0f}   boxes {env.load}   delivered {s['boxes_delivered']}", True, (240, 240, 240)), (10, 8))
        surf.blit(self.font.render(f"clinic stockout steps {s['stockout_steps']}", True, (240, 200, 200)), (10, 30))
        surf.blit(self.small.render(env.last_event.upper(), True, (255, 210, 80)), (10, 52))

        if self.mode == "human":
            if self.window is None:
                self.window = pygame.display.set_mode((w, h)); pygame.display.set_caption("Medicine Drone")
            self.window.blit(surf, (0, 0)); pygame.event.pump(); pygame.display.flip(); self.clock.tick(self.fps)
            return None
        return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), (1, 0, 2)).copy()


# ====================================================================== OpenGL renderer
def _col(c, a=1.0):
    gl.glColor4f(c[0] / 255, c[1] / 255, c[2] / 255, a)


def _rect(x, y, w, h, c, a=1.0):
    _col(c, a)
    gl.glBegin(gl.GL_QUADS)
    gl.glVertex2f(x, y); gl.glVertex2f(x + w, y); gl.glVertex2f(x + w, y + h); gl.glVertex2f(x, y + h)
    gl.glEnd()


def _poly(pts, c, a=1.0):
    _col(c, a)
    gl.glBegin(gl.GL_POLYGON)
    for px, py in pts:
        gl.glVertex2f(px, py)
    gl.glEnd()


def _circle(cx, cy, r, c, a=1.0, seg=28):
    _col(c, a)
    gl.glBegin(gl.GL_TRIANGLE_FAN)
    gl.glVertex2f(cx, cy)
    for i in range(seg + 1):
        t = 2 * math.pi * i / seg
        gl.glVertex2f(cx + r * math.cos(t), cy + r * math.sin(t))
    gl.glEnd()


def _ring(cx, cy, r, c, width=2.0, seg=28):
    _col(c)
    gl.glLineWidth(width)
    gl.glBegin(gl.GL_LINE_LOOP)
    for i in range(seg):
        t = 2 * math.pi * i / seg
        gl.glVertex2f(cx + r * math.cos(t), cy + r * math.sin(t))
    gl.glEnd()


def _line(p, q, c, width=2.0):
    _col(c)
    gl.glLineWidth(width)
    gl.glBegin(gl.GL_LINES)
    gl.glVertex2f(*p); gl.glVertex2f(*q)
    gl.glEnd()


class DroneRenderer:
    """OpenGL view. mode 'human' opens a window; 'rgb_array' renders in a hidden window and returns the pixels."""

    def __init__(self, mode, fps):
        pygame.init()
        self.mode, self.fps = mode, fps
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 15, bold=True)
        self.small = pygame.font.SysFont("monospace", 12, bold=True)
        self.window, self.soft, self.size = None, None, None

    # ---------- setup and teardown
    def _init_gl(self, w, h):
        if gl is None:
            raise RuntimeError("PyOpenGL is not installed")
        flags = pygame.OPENGL | pygame.DOUBLEBUF | (pygame.HIDDEN if self.mode != "human" else 0)
        try:
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
            self.window = pygame.display.set_mode((w, h), flags)
        except pygame.error:                                         # no anti-aliasing available
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 0)
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 0)
            self.window = pygame.display.set_mode((w, h), flags)
        pygame.display.set_caption("Medicine Drone (OpenGL)")
        gl.glViewport(0, 0, w, h)
        gl.glMatrixMode(gl.GL_PROJECTION); gl.glLoadIdentity(); gl.glOrtho(0, w, h, 0, -1, 1)   # pixel coordinates, y down
        gl.glMatrixMode(gl.GL_MODELVIEW); gl.glLoadIdentity()
        gl.glEnable(gl.GL_BLEND); gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_LINE_SMOOTH); gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        self.size = (w, h)

    def close(self):
        if self.soft is not None:
            self.soft.close()
        if self.window is not None:
            pygame.display.quit()
        pygame.quit()

    # ---------- helpers
    def _bar(self, x, y, w, frac, c, h=7):
        _rect(x, y, w, h, (40, 40, 40)); _rect(x, y, w * max(0.0, min(1.0, frac)), h, c)

    def _text(self, s, x, y, font, color):
        surf = font.render(s, True, color)
        tw, th = surf.get_size()
        tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, tw, th, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                        pygame.image.tostring(surf, "RGBA", False))
        gl.glEnable(gl.GL_TEXTURE_2D); gl.glColor4f(1, 1, 1, 1)
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0, 0); gl.glVertex2f(x, y); gl.glTexCoord2f(1, 0); gl.glVertex2f(x + tw, y)
        gl.glTexCoord2f(1, 1); gl.glVertex2f(x + tw, y + th); gl.glTexCoord2f(0, 1); gl.glVertex2f(x, y + th)
        gl.glEnd()
        gl.glDisable(gl.GL_TEXTURE_2D); gl.glDeleteTextures([tex])

    # ---------- the scene
    def draw(self, env):
        n = env.size
        w, h = n * CELL, n * CELL + TOP
        if self.window is None and self.soft is None:
            try:
                self._init_gl(w, h)
            except Exception as err:
                print(f"[rendering] OpenGL window not available ({err}); using the software renderer instead.")
                self.window = None
                self.soft = _SoftwareRenderer(self.mode, self.fps)
        if self.soft is not None:
            return self.soft.draw(env)

        # animation clock: real time in the window, simulation time for saved frames (so they are repeatable)
        anim = pygame.time.get_ticks() / 1000.0 if self.mode == "human" else env.t * 0.125
        gl.glClearColor(30 / 255, 34 / 255, 46 / 255, 1); gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        for x in range(n):                                              # countryside tiles
            for y in range(n):
                _rect(x * CELL, TOP + y * CELL, CELL, CELL, (112, 168, 92) if (x + y) % 2 == 0 else (104, 160, 86))
        for x in range(n):                                              # hills
            for y in range(n):
                if env.hills[x, y]:
                    cx, cy = x * CELL + CELL // 2, TOP + y * CELL + CELL // 2
                    _poly([(cx - 26, cy + 18), (cx, cy - 20), (cx + 26, cy + 18)], (128, 100, 70))
                    _poly([(cx - 10, cy + 2), (cx, cy - 20), (cx + 6, cy - 6)], (158, 128, 92))
        hx, hy = env.hub                                                # hub: depot with a red cross
        bx, by = hx * CELL + 4, TOP + hy * CELL + 6
        _rect(bx, by, CELL - 8, CELL - 12, (240, 240, 240))
        _rect(bx + 22, by + 8, 14, 34, (210, 40, 40)); _rect(bx + 12, by + 18, 34, 14, (210, 40, 40))
        self._text("HUB", bx + 17, by + CELL - 24, self.small, (20, 20, 20))
        for k in range(env.nc):                                         # clinics with stock bars
            px, py = env.cx[k] * CELL, TOP + env.cy[k] * CELL
            low = env.stock[k] < 0.2
            _rect(px + 10, py + 22, CELL - 20, CELL - 30, (200, 70, 60) if low else (230, 215, 180))
            _poly([(px + 6, py + 24), (px + CELL // 2, py + 6), (px + CELL - 6, py + 24)], (150, 60, 50))
            _rect(px + CELL // 2 - 6, py + 38, 12, 16, (60, 40, 30))
            self._bar(px + 8, py + 2, CELL - 16, env.stock[k], (230, 60, 50) if low else (80, 200, 110))
        for sx, sy in env.storms:                                       # storms, with a flickering bolt
            cx, cy = sx * CELL + CELL // 2, TOP + sy * CELL + CELL // 2
            for ox, oy, r in ((-14, 4, 16), (6, -2, 20), (18, 6, 14), (0, 10, 16)):
                _circle(cx + ox, cy + oy, r, (70, 74, 96))
            flicker = 0.65 + 0.35 * math.sin(anim * 14 + sx)
            _poly([(cx, cy + 6), (cx - 7, cy + 22), (cx - 1, cy + 22), (cx - 5, cy + 34), (cx + 8, cy + 16), (cx + 2, cy + 16)],
                  (250, 220, 60), flicker)

        gx, gy = env.dx * CELL + CELL // 2, TOP + env.dy * CELL + CELL // 2   # the drone
        gl.glPushMatrix(); gl.glTranslatef(gx, gy + 22, 0); gl.glScalef(1.0, 0.3, 1.0)
        _circle(0, 0, 14, (60, 90, 55), 0.7); gl.glPopMatrix()                # soft shadow
        for ox, oy in ((-16, -14), (16, -14), (-16, 12), (16, 12)):
            _line((gx, gy), (gx + ox, gy + oy), (50, 50, 60), 3)
            _circle(gx + ox, gy + oy, 8, (190, 190, 205), 0.55)
            gl.glPushMatrix(); gl.glTranslatef(gx + ox, gy + oy, 0); gl.glRotatef(anim * 1500 % 360, 0, 0, 1)
            _line((-10, 0), (10, 0), (60, 60, 70), 2.5); _line((0, -10), (0, 10), (60, 60, 70), 2.5)   # spinning blades
            gl.glPopMatrix()
            _ring(gx + ox, gy + oy, 8, (60, 60, 70), 2)
        _rect(gx - 9, gy - 9, 18, 18, (35, 90, 200))
        for b in range(env.load):
            _rect(gx - 8 + b * 9, gy + 14, 8, 8, (245, 245, 245)); _rect(gx - 5 + b * 9, gy + 15, 2, 6, (210, 40, 40))
        self._bar(gx - 18, gy - 26, 36, env.battery / env.max_battery, (250, 200, 50) if env.battery > 12 else (230, 60, 50))

        s = env.stats                                                   # status bar
        self._text(f"step {env.t}/{env.day_length}   battery {env.battery:4.0f}   boxes {env.load}   "
                   f"delivered {s['boxes_delivered']}", 10, 8, self.font, (240, 240, 240))
        self._text(f"clinic stockout steps {s['stockout_steps']}", 10, 30, self.font, (240, 200, 200))
        if env.last_event:
            self._text(env.last_event.upper(), 10, 52, self.small, (255, 210, 80))

        if self.mode == "human":
            pygame.event.pump(); pygame.display.flip(); self.clock.tick(self.fps)
            return None
        gl.glFinish()
        pix = gl.glReadPixels(0, 0, w, h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        pix = pix if isinstance(pix, (bytes, bytearray)) else np.asarray(pix).tobytes()
        return np.frombuffer(pix, dtype=np.uint8).reshape(h, w, 3)[::-1].copy()
