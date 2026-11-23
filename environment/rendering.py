"""Pygame renderer for Medicine Drone (retro top-down look, no image files needed)."""
import numpy as np
import pygame

CELL, TOP = 64, 70


class DroneRenderer:
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
