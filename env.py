"""
Self-Driving Car RL Environment
--------------------------------
A top-down 2D driving simulator featuring:
  * A 4-way intersection with lane markings
  * A traffic light that cycles GREEN -> YELLOW -> RED
  * Pedestrians that periodically cross the road
  * Other vehicles moving on the cross street
  * An agent car that must reach a goal (straight / left / right)

Observation (11-dim float vector):
  0 : agent x (normalized 0..1)
  1 : agent y (normalized 0..1)
  2 : heading  (0=HDG_N, 1=HDG_E, 2=HDG_S, 3=HDG_W) / 3
  3 : speed    (0..MAX_SPEED) / MAX_SPEED
  4 : traffic light  (0=green, 1=yellow, 2=red) / 2
  5 : distance to stop line ahead (normalized)
  6 : pedestrian on crosswalk ahead (0/1)
  7 : vehicle ahead within 4 cells (0/1)
  8 : vehicle on crossing lane near intersection (0/1)
  9 : goal direction relative to heading (-1 left, 0 straight, 1 right) mapped to 0..1
 10 : distance to goal (normalized)

Actions (Discrete 5):
  0: NOOP (maintain)
  1: ACCELERATE
  2: BRAKE / STOP
  3: TURN LEFT  (only valid inside intersection)
  4: TURN RIGHT (only valid inside intersection)

Reward:
  +1.0   step where speed>0 and heading aligned with goal path
  -0.05  per timestep (efficiency)
  -50    collision with pedestrian
  -40    collision with vehicle
  -25    running a red light (entering intersection on red)
  -5     illegal turn attempt
  +150   reaching goal
"""
from __future__ import annotations
import os
import math
import random
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np
import pygame

# ---------------- Configuration ----------------
GRID = 20                 # 20x20 cells
CELL = 32                 # pixel size per cell
W = H = GRID * CELL       # 640x640
FPS = 30
MAX_SPEED = 2             # cells per step
MAX_STEPS = 400

# Road layout: two perpendicular 2-lane roads intersecting at center
ROAD_ROWS = (GRID // 2 - 1, GRID // 2)   # horizontal road lanes (y indices)
ROAD_COLS = (GRID // 2 - 1, GRID // 2)   # vertical road lanes (x indices)
INTERSECTION = {(c, r) for c in ROAD_COLS for r in ROAD_ROWS}
STOP_LINE_Y = ROAD_ROWS[1] + 1   # south-bound agent stops here (agent drives north)

# Headings (avoid single-letter names that clash with W/H canvas size)
HDG_N, HDG_E, HDG_S, HDG_W = 0, 1, 2, 3
HEAD_DX = {HDG_N: 0, HDG_E: 1, HDG_S: 0, HDG_W: -1}
HEAD_DY = {HDG_N: -1, HDG_E: 0, HDG_S: 1, HDG_W: 0}

# Colors
C_BG = (34, 139, 63)
C_ROAD = (55, 55, 60)
C_LANE = (240, 220, 90)
C_CROSS = (235, 235, 235)
C_AGENT = (40, 130, 240)
C_VEH = (200, 60, 60)
C_PED = (250, 240, 240)
C_LIGHT_BG = (25, 25, 25)
C_GREEN = (60, 220, 90)
C_YELLOW = (240, 200, 40)
C_RED = (230, 60, 60)
C_TEXT = (255, 255, 255)

@dataclass
class Vehicle:
    x: int
    y: int
    heading: int
    speed: int = 1
    color: Tuple[int, int, int] = C_VEH

@dataclass
class Pedestrian:
    x: float
    y: float
    dx: float
    dy: float
    active: bool = True


class DrivingEnv:
    """Grid-based self-driving environment with pygame rendering."""

    action_space_n = 5
    observation_dim = 11

    def __init__(self, render_mode: Optional[str] = None, seed: Optional[int] = None):
        self.render_mode = render_mode
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self._pygame_ready = False
        self.screen: Optional[pygame.Surface] = None
        self.font: Optional[pygame.font.Font] = None
        self.big_font: Optional[pygame.font.Font] = None

        self.reset()

    # ---------- Lifecycle ----------
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)

        # Agent starts at the bottom of the vertical road, heading North
        self.agent = Vehicle(
            x=ROAD_COLS[0], y=GRID - 1, heading=HDG_N, speed=1, color=C_AGENT
        )

        # Goal: 0=straight (top), 1=left (west end), 2=right (east end)
        self.goal_type = self.rng.choice([0, 1, 2])
        if self.goal_type == 0:
            self.goal = (ROAD_COLS[0], 0)
        elif self.goal_type == 1:
            self.goal = (0, ROAD_ROWS[0])
        else:
            self.goal = (GRID - 1, ROAD_ROWS[0])

        # Traffic light cycle: GREEN(40) -> YELLOW(10) -> RED(40)
        self.light_timer = self.rng.randint(0, 89)

        # Other vehicles on horizontal road
        self.vehicles: List[Vehicle] = []
        self._spawn_vehicle(force=True)

        # Pedestrians on crosswalks
        self.pedestrians: List[Pedestrian] = []

        self.steps = 0
        self.done = False
        self.last_reward = 0.0
        self.info_msg = ""
        self.total_reward = 0.0
        self.turned = False  # whether agent has completed its turn
        return self._obs()

    # ---------- Traffic light ----------
    @property
    def light_state(self) -> int:
        """0=green, 1=yellow, 2=red (for the agent facing North)."""
        t = self.light_timer % 90
        if t < 40:
            return 0
        if t < 50:
            return 1
        return 2

    # ---------- Spawns ----------
    def _spawn_vehicle(self, force: bool = False):
        if not force and self.rng.random() > 0.05:
            return
        if len(self.vehicles) >= 3:
            return
        # Spawn on horizontal road, random direction
        if self.rng.random() < 0.5:
            v = Vehicle(x=0, y=ROAD_ROWS[1], heading=HDG_E, speed=1)
        else:
            v = Vehicle(x=GRID - 1, y=ROAD_ROWS[0], heading=HDG_W, speed=1)
        # avoid stacking
        for other in self.vehicles:
            if other.x == v.x and other.y == v.y:
                return
        self.vehicles.append(v)

    def _spawn_pedestrian(self):
        if self.rng.random() > 0.03:
            return
        if len(self.pedestrians) >= 2:
            return
        # Cross vertical road horizontally near intersection top or bottom
        side = self.rng.choice(["top", "bottom"])
        y = ROAD_ROWS[0] - 1 if side == "top" else ROAD_ROWS[1] + 1
        if self.rng.random() < 0.5:
            p = Pedestrian(x=ROAD_COLS[0] - 0.5, y=y + 0.5, dx=0.25, dy=0.0)
        else:
            p = Pedestrian(x=ROAD_COLS[1] + 1.5, y=y + 0.5, dx=-0.25, dy=0.0)
        self.pedestrians.append(p)

    # ---------- Step ----------
    def step(self, action: int):
        assert not self.done, "Call reset()"
        reward = -0.05
        self.info_msg = ""

        # 1) Apply action
        prev_heading = self.agent.heading
        in_intersection = (self.agent.x, self.agent.y) in INTERSECTION

        if action == 1:  # accelerate
            self.agent.speed = min(MAX_SPEED, self.agent.speed + 1)
        elif action == 2:  # brake
            self.agent.speed = max(0, self.agent.speed - 1)
        elif action == 3:  # turn left
            if in_intersection and not self.turned:
                self.agent.heading = (self.agent.heading - 1) % 4
                self.turned = True
            else:
                reward -= 5
                self.info_msg = "Illegal turn"
        elif action == 4:  # turn right
            if in_intersection and not self.turned:
                self.agent.heading = (self.agent.heading + 1) % 4
                self.turned = True
            else:
                reward -= 5
                self.info_msg = "Illegal turn"

        # 2) Red-light check: entering intersection from south on red
        if (
            self.agent.speed > 0
            and self.agent.heading == HDG_N
            and self.agent.y == STOP_LINE_Y
            and self.light_state == 2
        ):
            reward -= 25
            self.info_msg = "Ran red light!"

        # 3) Move agent
        for _ in range(self.agent.speed):
            nx = self.agent.x + HEAD_DX[self.agent.heading]
            ny = self.agent.y + HEAD_DY[self.agent.heading]
            if 0 <= nx < GRID and 0 <= ny < GRID:
                self.agent.x, self.agent.y = nx, ny

        # 4) Move other vehicles
        for v in list(self.vehicles):
            v.x += HEAD_DX[v.heading] * v.speed
            v.y += HEAD_DY[v.heading] * v.speed
            if not (0 <= v.x < GRID and 0 <= v.y < GRID):
                self.vehicles.remove(v)
        self._spawn_vehicle()

        # 5) Move pedestrians
        for p in list(self.pedestrians):
            p.x += p.dx
            p.y += p.dy
            if not (-1 <= p.x <= GRID + 1):
                self.pedestrians.remove(p)
        self._spawn_pedestrian()

        # 6) Advance traffic light
        self.light_timer += 1

        # 7) Collisions
        for v in self.vehicles:
            if v.x == self.agent.x and v.y == self.agent.y:
                reward -= 40
                self.info_msg = "Crashed into vehicle!"
                self.done = True
        for p in self.pedestrians:
            if abs(p.x - self.agent.x) < 0.7 and abs(p.y - self.agent.y) < 0.7:
                reward -= 50
                self.info_msg = "Hit a pedestrian!"
                self.done = True

        # 8) Progress reward: getting closer to goal
        dist = abs(self.agent.x - self.goal[0]) + abs(self.agent.y - self.goal[1])
        if not hasattr(self, "_prev_dist"):
            self._prev_dist = dist
        if dist < self._prev_dist:
            reward += 1.0
        self._prev_dist = dist

        # 9) Goal reached
        if (self.agent.x, self.agent.y) == self.goal:
            reward += 150
            self.info_msg = "Goal reached!"
            self.done = True

        # 10) Step limit
        self.steps += 1
        if self.steps >= MAX_STEPS:
            self.done = True

        self.last_reward = reward
        self.total_reward += reward
        return self._obs(), reward, self.done, {"msg": self.info_msg}

    # ---------- Observation ----------
    def _obs(self) -> np.ndarray:
        ax, ay = self.agent.x, self.agent.y
        # distance to stop line ahead (only meaningful when heading HDG_N)
        if self.agent.heading == HDG_N:
            d_stop = max(0, ay - STOP_LINE_Y) / GRID
        else:
            d_stop = 1.0

        # pedestrian ahead within 3 cells in current heading
        ped_ahead = 0.0
        for p in self.pedestrians:
            fx = ax + HEAD_DX[self.agent.heading] * 2
            fy = ay + HEAD_DY[self.agent.heading] * 2
            if abs(p.x - fx) < 1.5 and abs(p.y - fy) < 1.5:
                ped_ahead = 1.0

        # vehicle ahead within 4 cells
        veh_ahead = 0.0
        for v in self.vehicles:
            for k in range(1, 5):
                if v.x == ax + HEAD_DX[self.agent.heading] * k and v.y == ay + HEAD_DY[self.agent.heading] * k:
                    veh_ahead = 1.0

        # vehicle near intersection on crossing lane
        veh_cross = 0.0
        for v in self.vehicles:
            if v.y in ROAD_ROWS and abs(v.x - GRID // 2) <= 3:
                veh_cross = 1.0

        # goal direction relative to heading
        # 0=straight, 1=left, 2=right -> encode as 0.0/0.5/1.0
        gmap = {0: 0.5, 1: 0.0, 2: 1.0}
        goal_dir = gmap[self.goal_type]

        d_goal = (abs(ax - self.goal[0]) + abs(ay - self.goal[1])) / (2 * GRID)

        return np.array([
            ax / GRID,
            ay / GRID,
            self.agent.heading / 3.0,
            self.agent.speed / MAX_SPEED,
            self.light_state / 2.0,
            d_stop,
            ped_ahead,
            veh_ahead,
            veh_cross,
            goal_dir,
            d_goal,
        ], dtype=np.float32)

    # ---------- Rendering ----------
    def _ensure_pygame(self):
        if self._pygame_ready:
            return
        if self.render_mode == "rgb_array":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        pygame.font.init()
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption("Self-Driving RL")
        else:
            self.screen = pygame.Surface((W, H))
        self.font = pygame.font.SysFont("arial", 14, bold=True)
        self.big_font = pygame.font.SysFont("arial", 22, bold=True)
        self._pygame_ready = True

    def render(self) -> np.ndarray:
        self._ensure_pygame()
        s = self.screen
        s.fill(C_BG)

        # Roads
        pygame.draw.rect(s, C_ROAD, (0, ROAD_ROWS[0] * CELL, W, 2 * CELL))
        pygame.draw.rect(s, C_ROAD, (ROAD_COLS[0] * CELL, 0, 2 * CELL, H))

        # Lane markings (dashed center)
        for i in range(0, W, 20):
            if not (ROAD_COLS[0] * CELL <= i <= ROAD_COLS[1] * CELL + CELL):
                pygame.draw.rect(s, C_LANE, (i, (ROAD_ROWS[0] + 1) * CELL - 2, 10, 4))
        for j in range(0, H, 20):
            if not (ROAD_ROWS[0] * CELL <= j <= ROAD_ROWS[1] * CELL + CELL):
                pygame.draw.rect(s, C_LANE, ((ROAD_COLS[0] + 1) * CELL - 2, j, 4, 10))

        # Crosswalks (zebra) around intersection
        for k in range(6):
            pygame.draw.rect(s, C_CROSS,
                (ROAD_COLS[0] * CELL + k * 12, (ROAD_ROWS[1] + 1) * CELL, 8, 6))
            pygame.draw.rect(s, C_CROSS,
                (ROAD_COLS[0] * CELL + k * 12, ROAD_ROWS[0] * CELL - 6, 8, 6))

        # Stop line for agent
        pygame.draw.rect(s, (255, 255, 255),
            (ROAD_COLS[0] * CELL, (STOP_LINE_Y) * CELL - 3, CELL, 3))

        # Traffic light box
        lx, ly = (ROAD_COLS[0] - 1) * CELL - 6, (ROAD_ROWS[1] + 1) * CELL + 10
        pygame.draw.rect(s, C_LIGHT_BG, (lx, ly, 20, 56), border_radius=4)
        colors = [C_RED, C_YELLOW, C_GREEN]
        active = [2, 1, 0].index(self.light_state)  # top=red
        for i, c in enumerate(colors):
            col = c if i == active else (60, 60, 60)
            pygame.draw.circle(s, col, (lx + 10, ly + 10 + i * 18), 7)

        # Pedestrians
        for p in self.pedestrians:
            pygame.draw.circle(s, C_PED,
                (int(p.x * CELL + CELL / 2), int(p.y * CELL + CELL / 2)), 7)
            pygame.draw.circle(s, (0, 0, 0),
                (int(p.x * CELL + CELL / 2), int(p.y * CELL + CELL / 2)), 7, 1)

        # Other vehicles
        for v in self.vehicles:
            self._draw_car(v.x, v.y, v.heading, v.color)

        # Goal marker
        gx, gy = self.goal
        pygame.draw.rect(s, (255, 220, 60),
            (gx * CELL + 6, gy * CELL + 6, CELL - 12, CELL - 12), 2)
        s.blit(self.font.render("GOAL", True, (255, 220, 60)),
               (gx * CELL + 2, gy * CELL - 14))

        # Agent
        self._draw_car(self.agent.x, self.agent.y, self.agent.heading, C_AGENT)

        # HUD
        goal_txt = {0: "Straight", 1: "Left", 2: "Right"}[self.goal_type]
        light_txt = {0: "GREEN", 1: "YELLOW", 2: "RED"}[self.light_state]
        hud = f"Step {self.steps}  Reward {self.total_reward:+.1f}  Light {light_txt}  Goal {goal_txt}"
        s.blit(self.font.render(hud, True, C_TEXT), (8, 6))
        if self.info_msg:
            s.blit(self.big_font.render(self.info_msg, True, (255, 200, 60)), (8, H - 30))

        if self.render_mode == "human":
            pygame.display.flip()

        arr = pygame.surfarray.array3d(s)
        return np.transpose(arr, (1, 0, 2))

    def _draw_car(self, x, y, heading, color):
        cx, cy = x * CELL + CELL // 2, y * CELL + CELL // 2
        if heading in (HDG_N, HDG_S):
            rect = pygame.Rect(0, 0, CELL - 10, CELL - 4)
        else:
            rect = pygame.Rect(0, 0, CELL - 4, CELL - 10)
        rect.center = (cx, cy)
        pygame.draw.rect(self.screen, color, rect, border_radius=4)
        pygame.draw.rect(self.screen, (10, 10, 20), rect, 1, border_radius=4)
        # windshield indicator
        dx, dy = HEAD_DX[heading] * 6, HEAD_DY[heading] * 6
        pygame.draw.circle(self.screen, (240, 240, 255), (cx + dx, cy + dy), 3)

    def close(self):
        if self._pygame_ready:
            pygame.quit()
            self._pygame_ready = False
