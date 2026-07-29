"""Train DQN agent on DrivingEnv.

Default increased to 1500 episodes because the new reward shaping in env.py
needs more experience to converge on stopping/yielding behaviour.
Usage:  python train.py [episodes]
"""
from __future__ import annotations
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from env import DrivingEnv
from dqn import DQNAgent


def train(episodes: int = 1500, save_path: str = "dqn_car.pt", log_every: int = 25):
    env = DrivingEnv(seed=0)
    agent = DQNAgent(env.observation_dim, env.action_space_n)

    rewards, successes, red_runs, crashes = [], [], [], []
    t0 = time.time()
    for ep in range(1, episodes + 1):
        s = env.reset(seed=ep)
        total = 0.0
        success = ran_red = crashed = False
        while True:
            a = agent.act(s)
            s2, r, done, info = env.step(a)
            agent.remember(s, a, r, s2, done)
            agent.train_step()
            s = s2
            total += r
            msg = info.get("msg", "")
            if msg == "Goal reached!":
                success = True
            if "RED" in msg:
                ran_red = True
            if "Crashed" in msg or "Hit" in msg:
                crashed = True
            if done:
                break
        agent.decay_eps()
        rewards.append(total)
        successes.append(1 if success else 0)
        red_runs.append(1 if ran_red else 0)
        crashes.append(1 if crashed else 0)
        if ep % log_every == 0:
            print(
                f"ep {ep:4d} | R {np.mean(rewards[-log_every:]):+7.1f}"
                f" | success {np.mean(successes[-log_every:])*100:5.1f}%"
                f" | redRun {np.mean(red_runs[-log_every:])*100:4.1f}%"
                f" | crash  {np.mean(crashes[-log_every:])*100:4.1f}%"
                f" | eps {agent.eps:.3f} | {time.time()-t0:.1f}s"
            )

    agent.save(save_path)
    print(f"Saved policy -> {save_path}")
    return agent


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    train(n)
