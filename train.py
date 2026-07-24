"""Train DQN agent on DrivingEnv."""
from __future__ import annotations
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from env import DrivingEnv
from dqn import DQNAgent


def train(episodes: int = 400, save_path: str = "dqn_car.pt", log_every: int = 20):
    env = DrivingEnv(seed=0)
    agent = DQNAgent(env.observation_dim, env.action_space_n)

    rewards, successes = [], []
    t0 = time.time()
    for ep in range(1, episodes + 1):
        s = env.reset(seed=ep)
        total = 0.0
        success = False
        while True:
            a = agent.act(s)
            s2, r, done, info = env.step(a)
            agent.remember(s, a, r, s2, done)
            agent.train_step()
            s = s2
            total += r
            if info.get("msg") == "Goal reached!":
                success = True
            if done:
                break
        agent.decay_eps()
        rewards.append(total)
        successes.append(1 if success else 0)
        if ep % log_every == 0:
            avg = np.mean(rewards[-log_every:])
            sr = np.mean(successes[-log_every:])
            print(f"ep {ep:4d} | reward {avg:+7.1f} | success {sr*100:5.1f}% | eps {agent.eps:.3f} | {time.time()-t0:.1f}s")

    agent.save(save_path)
    print(f"Saved policy -> {save_path}")
    return agent


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    train(n)
