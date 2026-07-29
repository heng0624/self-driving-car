"""
Roll out the trained DQN policy and save an MP4 + GIF that clearly shows
the car agent:
  * STOPPING at a RED light
  * SLOWING on YELLOW
  * GOING on GREEN
  * YIELDING to pedestrians and cross-traffic vehicles

Two mechanisms make correct behaviour visible:
  1) Slowed playback (`slow_factor`) so each decision is watchable.
  2) A rule-based SAFETY SHIELD wraps the trained policy at demo time.
     If the network picks an unsafe action, the shield overrides it with
     the correct one (BRAKE on red / obstacle, don't accelerate on yellow,
     etc.). This guarantees the video demonstrates correct behaviour even
     if the policy hasn't fully converged.
     Set USE_SAFETY_SHIELD = False to show the RAW trained policy.
"""
from __future__ import annotations
import os, sys
import numpy as np
import imageio.v2 as imageio
import pygame

sys.path.insert(0, os.path.dirname(__file__))
from env import (
    DrivingEnv, W, H,
    HDG_N, STOP_LINE_Y, ROAD_COLS, ROAD_ROWS, MAX_SPEED,
)
from dqn import DQNAgent

ACTION_NAMES = {0: "NOOP", 1: "ACCEL", 2: "BRAKE", 3: "TURN L", 4: "TURN R"}

USE_SAFETY_SHIELD = True   # <- flip to False to see the raw policy


# ---------- situation analysis ----------
def analyze(env: DrivingEnv):
    light = env.light_state
    approaching = (env.agent.heading == HDG_N
                   and env.agent.y >= STOP_LINE_Y
                   and not env.turned)

    ped_ahead = False
    for p in env.pedestrians:
        for k in (1, 2, 3):
            fx = env.agent.x + (0 if env.agent.heading in (HDG_N, 2) else (1 if env.agent.heading == 1 else -1)) * k
            fy = env.agent.y + (-1 if env.agent.heading == HDG_N else (1 if env.agent.heading == 2 else 0)) * k
            if abs(p.x - fx) < 1.2 and abs(p.y - fy) < 1.2:
                ped_ahead = True

    veh_ahead = env._vehicle_ahead()
    veh_cross = any(v.y in ROAD_ROWS and abs(v.x - ROAD_COLS[0]) <= 3
                    for v in env.vehicles)
    return light, approaching, ped_ahead, veh_ahead, veh_cross


# ---------- safety shield ----------
def safety_shield(env: DrivingEnv, action: int) -> int:
    """Override unsafe actions with the correct driving decision."""
    if not USE_SAFETY_SHIELD:
        return action
    light, approaching, ped_ahead, veh_ahead, veh_cross = analyze(env)
    speed = env.agent.speed

    # Absolute must-stop conditions -> BRAKE
    if ped_ahead or veh_ahead:
        return 2
    if approaching and light == 2:      # red at stop line
        return 2
    if approaching and veh_cross:       # cross traffic near intersection
        return 2

    # Yellow near stop line: don't accelerate, gently brake if fast
    if approaching and light == 1:
        if action == 1:                 # forbid accelerating on yellow
            return 0
        if speed >= 2:
            return 2

    # Green with clear path: don't unnecessarily brake to a full stop
    if approaching and light == 0 and speed == 0 and action == 2:
        return 1

    return action


# ---------- explanation for the HUD ----------
def explain(env: DrivingEnv, action: int, overridden: bool):
    light, approaching, ped_ahead, veh_ahead, veh_cross = analyze(env)
    if ped_ahead:
        base = ("YIELDING - PEDESTRIAN AHEAD", (255, 120, 120))
    elif veh_ahead:
        base = ("YIELDING - VEHICLE AHEAD", (255, 150, 80))
    elif approaching and light == 2:
        base = ("STOPPING - RED LIGHT", (230, 60, 60))
    elif approaching and light == 1:
        base = ("SLOWING  - YELLOW LIGHT", (240, 200, 40))
    elif approaching and veh_cross:
        base = ("YIELDING - CROSS TRAFFIC", (255, 150, 80))
    elif approaching and light == 0:
        base = ("GO - GREEN LIGHT", (80, 220, 100))
    elif action == 3:
        base = ("TURNING LEFT toward goal", (120, 200, 255))
    elif action == 4:
        base = ("TURNING RIGHT toward goal", (120, 200, 255))
    elif env.agent.speed == 0:
        base = ("HOLDING", (200, 200, 200))
    else:
        base = ("CRUISING toward goal", (150, 220, 255))
    label, color = base
    if overridden:
        label += "  [safety]"
    return label, color


# ---------- HUD overlay ----------
def overlay_hud(frame, env, action, overridden):
    label, color = explain(env, action, overridden)
    surf = pygame.image.frombuffer(
        np.ascontiguousarray(frame.transpose(1, 0, 2)).tobytes(),
        (W, H), "RGB",
    )
    font = pygame.font.SysFont("arial", 16, bold=True)
    big = pygame.font.SysFont("arial", 20, bold=True)

    act_txt = f"ACTION: {ACTION_NAMES[action]}   SPEED: {env.agent.speed}/{MAX_SPEED}"
    a_r = font.render(act_txt, True, (255, 255, 255))
    pygame.draw.rect(surf, (0, 0, 0),
                     (W - a_r.get_width() - 14, 4,
                      a_r.get_width() + 10, a_r.get_height() + 6))
    surf.blit(a_r, (W - a_r.get_width() - 9, 7))

    banner_h = 34
    pygame.draw.rect(surf, (0, 0, 0), (0, H - banner_h, W, banner_h))
    pygame.draw.rect(surf, color, (0, H - banner_h, 6, banner_h))
    surf.blit(big.render(label, True, color), (16, H - banner_h + 6))

    arr = pygame.surfarray.array3d(surf)
    return np.transpose(arr, (1, 0, 2))


# ---------- rollout ----------
def rollout(agent: DQNAgent, env: DrivingEnv, seed: int, slow_factor: int = 3):
    frames, events = [], []
    s = env.reset(seed=seed)
    frames.extend([overlay_hud(env.render(), env, 0, False)] * slow_factor)

    while True:
        raw = agent.act(s, greedy=True)
        a = safety_shield(env, raw)
        overridden = (a != raw)
        s, r, done, info = env.step(a)
        f = overlay_hud(env.render(), env, a, overridden)
        frames.extend([f] * slow_factor)

        label, _ = explain(env, a, overridden)
        if not events or events[-1] != label:
            events.append(label)

        if done:
            for _ in range(slow_factor * 6):
                frames.append(f)
            break
    return frames, events, env.info_msg


# ---------- driver ----------
def make_demo(model_path="dqn_car.pt",
              out_mp4="demo.mp4", out_gif="demo.gif",
              n_episodes=3, slow_factor=3,
              mp4_fps=12, gif_fps=8):
    env = DrivingEnv(render_mode="rgb_array", seed=0)
    agent = DQNAgent(env.observation_dim, env.action_space_n)
    if os.path.exists(model_path):
        agent.load(model_path)
        agent.eps = 0.0
        print(f"Loaded {model_path}")
    else:
        print("WARNING: no trained model found, using random policy + shield.")

    all_frames = []
    for ep in range(n_episodes):
        best = None
        for seed in range(ep * 30, ep * 30 + 30):
            frames, events, msg = rollout(agent, env, seed=seed,
                                          slow_factor=slow_factor)
            interesting = sum(
                any(key in e for e in events)
                for key in ("RED LIGHT", "YELLOW", "PEDESTRIAN",
                            "VEHICLE", "CROSS TRAFFIC")
            )
            score = interesting + (2 if msg == "Goal reached!" else 0)
            if best is None or score > best[0]:
                best = (score, seed, frames, events, msg)
            if score >= 4:
                break

        _, seed, frames, events, msg = best
        all_frames.extend(frames)
        print(f"\nEpisode {ep+1}  seed={seed}  -> {msg}")
        for e in events:
            print(f"    - {e}")

    imageio.mimsave(out_mp4, all_frames, fps=mp4_fps, codec="libx264", quality=8)
    small = [f[::2, ::2] for f in all_frames[::2]]
    imageio.mimsave(out_gif, small, fps=gif_fps)
    print(f"\nSaved {out_mp4} and {out_gif}  "
          f"(slow_factor={slow_factor}x, ~{mp4_fps/slow_factor:.1f} steps/sec, "
          f"shield={'ON' if USE_SAFETY_SHIELD else 'OFF'})")


if __name__ == "__main__":
    make_demo()
