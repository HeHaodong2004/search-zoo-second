# domains_cc/path_planning_rl/train/train_ppo.py
# -*- coding: utf-8 -*-
import argparse
import yaml
from pathlib import Path
import imageio

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import random
from domains_cc.map_and_scen_utils import parse_map_file, parse_scen_file

from domains_cc.path_planning_rl.env.planning_env import PathPlanningMaskEnv



class CustomEvalCallback(BaseCallback):
    """
    每隔 eval_freq 步:
      1) 在 eval_env (VecMonitor) 上跑 n_eval_episodes 次评估: 统计 avg reward, avg length, success rate
      2) 用单 env 跑一条 deterministic rollout, 存成 GIF
      3) 如果 avg reward >= reward_threshold, 终止训练
    """
    def __init__(
        self,
        eval_env,
        env_kwargs: dict,
        gif_dir: str,
        eval_freq: int,
        reward_threshold: float,
        n_eval_episodes: int = 5,
        fps: int = 10,
        prefix: str = "eval",
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.env_kwargs = env_kwargs
        self.gif_dir = Path(gif_dir)
        self.gif_dir.mkdir(parents=True, exist_ok=True)
        self.eval_freq = eval_freq
        self.reward_threshold = reward_threshold
        self.n_eval_episodes = n_eval_episodes
        self.fps = fps
        self.prefix = prefix
        self.count = 0

    def _on_step(self) -> bool:
        # 只有在走够 eval_freq 才执行评估
        if self.num_timesteps % self.eval_freq == 0:
            # ---- 1) VecEnv 评估 ----
            rewards, lengths, successes = [], [], 0
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = np.zeros((self.eval_env.num_envs,), dtype=bool)
                total_r, steps, info = 0.0, 0, {}
                while not done.all():
                    action, _ = self.model.predict(
                        obs, deterministic=True,
                        action_masks=obs["action_mask"]
                    )
                    obs, rews, dones, infos = self.eval_env.step(action)
                    total_r += rews[0]
                    steps  += 1
                    done = dones
                    info = infos[0]
                rewards.append(total_r)
                lengths.append(steps)
                if info.get("reached", False):
                    successes += 1

            mean_r = float(np.mean(rewards))
            std_r  = float(np.std(rewards))
            mean_l = float(np.mean(lengths))
            succ_rate = successes / self.n_eval_episodes
            print(
                f"[Eval] episodes={self.n_eval_episodes}  "
                f"mean_reward={mean_r:.2f}±{std_r:.2f}  "
                f"mean_length={mean_l:.1f}  "
                f"success_rate={succ_rate:.0%}"
            )

            # 如果达到了阈值，就停止训练
            if mean_r >= self.reward_threshold:
                print(f"[Eval] mean_reward ≥ {self.reward_threshold}, stopping training.")
                return False

            # ---- 2) 单 env 生成 GIF ----
            demo_env = make_maskable_env(
                scen_files       = self.env_kwargs["scen_files"],
                dynamics_config  = self.env_kwargs["dynamics_config"],
                footprint_config = self.env_kwargs["footprint_config"],
            )
            reset_out = demo_env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            frames, done = [], False
            while not done:
                action, _ = self.model.predict(
                    obs, deterministic=True,
                    action_masks=obs["action_mask"]
                )
                step_out = demo_env.step(action)
                if len(step_out) == 5:
                    obs, _, terminated, truncated, _ = step_out
                    done = terminated or truncated
                else:
                    obs, _, done, _ = step_out
                frames.append(demo_env.env.render(mode="rgb_array"))

            gif_path = self.gif_dir / f"{self.prefix}_{self.count:03d}.gif"
            imageio.mimsave(str(gif_path), frames, fps=self.fps)
            print(f"[EvalAndGif] saved {gif_path}")
            self.count += 1

        return True


def make_maskable_env(scen_files, dynamics_config, footprint_config):
    # 随机选一个场景文件
    scen = random.choice(scen_files)
    # 解析出这个场景一共有多少个起终点对
    pairs = parse_scen_file(scen)
    idx   = random.randrange(len(pairs))
    base_env = PathPlanningMaskEnv(
        scen_file      = scen,
        problem_index  = idx,
        dynamics_config= dynamics_config,
        footprint_config=footprint_config,
    )
    return ActionMasker(base_env, lambda e: e.action_masks())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="ppo_config.yaml 的路径")
    parser.add_argument("--logdir", default="logs", help="输出目录")
    args = parser.parse_args()

    # 1) 读取配置
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    env_kwargs = dict(
        scen_files        = cfg["scen_files"],
        #problem_index    = int(cfg.get("problem_index", 0)),
        dynamics_config  = cfg["dynamics_config"],
        footprint_config = cfg["footprint_config"],
    )
    total_timesteps   = int(cfg.get("total_timesteps", 100_000))
    eval_freq         = int(cfg.get("eval_freq",     10_000))
    reward_threshold  = float(cfg.get("reward_threshold",   10.0))
    n_eval_episodes   = int(cfg.get("n_eval_episodes",       5))

    # 2) 输出目录
    logdir   = Path(args.logdir)
    tb_dir   = logdir / "tensorboard"
    best_dir = logdir / "best_model"
    gif_dir  = logdir / "gifs"
    for d in (logdir, tb_dir, best_dir, gif_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 3) 向量化 + Maskable
    def _make(): return make_maskable_env(**env_kwargs)
    train_env = VecMonitor(DummyVecEnv([_make]))
    eval_env  = VecMonitor(DummyVecEnv([_make]))

    # 4) 自定义评估回调
    eval_cb = CustomEvalCallback(
        eval_env=eval_env,
        env_kwargs=env_kwargs,
        gif_dir=str(gif_dir),
        eval_freq=eval_freq,
        reward_threshold=reward_threshold,
        n_eval_episodes=n_eval_episodes,
        fps=10,
        prefix="eval",
        verbose=1
    )

    # 5) MaskablePPO
    model = MaskablePPO(
        policy          = "MultiInputPolicy",
        env             = train_env,
        learning_rate   = cfg.get("learning_rate", 3e-4),
        n_steps         = cfg.get("n_steps",       2048),
        batch_size      = cfg.get("batch_size",     64),
        n_epochs        = cfg.get("n_epochs",       10),
        gamma           = cfg.get("gamma",        0.99),
        clip_range      = cfg.get("clip_range",    0.2),
        ent_coef        = cfg.get("ent_coef",      0.0),
        tensorboard_log = str(tb_dir),
        verbose         = 1,
    )

    # 6) 训练，只传入我们的 eval_cb
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_cb
    )

    model.save(str(logdir / "final_model"))
    print("✅ Training complete. Outputs saved in", args.logdir)


if __name__ == "__main__":
    main()
