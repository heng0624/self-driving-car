"""Roll out trained policy and save an MP4 + GIF."""
from __future__ import annotations
import os, sys
import numpy as np
import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(__file__))
from env import DrivingEnv
from dqn import DQNAgent


def rollout(agent: DQNAgent, env: DrivingEnv, seed: int) -> list:
    frames = []
    s = env.reset(seed=seed)
    frames.append(env.render())
    while True:
        a = agent.act(s, greedy=True)
        s, r, done, _info = env.step(a)
        frames.append(env.render())
        if done:
            break
    return frames


def make_demo(model_path: str = "dqn_car.pt",
              out_mp4: str = "demo.mp4",
              out_gif: str = "demo.gif",
              n_episodes: int = 3):
    env = DrivingEnv(render_mode="rgb_array", seed=42)
    agent = DQNAgent(env.observation_dim, env.action_space_n)
    if os.path.exists(model_path):
        agent.load(model_path)
        agent.eps = 0.0
        print(f"Loaded {model_path}")
    else:
        print("No model file; using random policy")

    all_frames = []
    for ep in range(n_episodes):
        # try a few seeds to get a nice successful demo
        for seed in range(ep * 10, ep * 10 + 6):
            frames = rollout(agent, env, seed=seed)
            success = env.info_msg == "Goal reached!"
            all_frames.extend(frames)
            print(f"episode {ep+1} seed {seed}: {len(frames)} steps, {env.info_msg}")
            if success:
                break

    # Save mp4 and gif
    imageio.mimsave(out_mp4, all_frames, fps=12, codec="libx264", quality=8)
    small = [f[::2, ::2] for f in all_frames[::2]]
    imageio.mimsave(out_gif, small, fps=8)
    print(f"Saved {out_mp4} and {out_gif}")


if __name__ == "__main__":
    make_demo()
