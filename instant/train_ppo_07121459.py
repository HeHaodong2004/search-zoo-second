# domains_cc/path_planning_rl/train/train_ppo.py
# -*- coding: utf-8 -*-
import os
import argparse
import yaml
from pathlib import Path
import imageio

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from domains_cc.path_planning_rl.env.planning_env import PathPlanningMaskEnv

class EvalAndGifCallback(EvalCallback):
    """
    继承自 EvalCallback：每当 eval 完成后，构造一个新的 MaskedEnv
    跑一条 deterministic rollout，保存成 GIF。
    """
    def __init__(
        self,
        env_kwargs: dict,
        gif_dir: str,
        fps: int = 10,
        prefix: str = "eval",
        **eval_kwargs
    ):
        super().__init__(**eval_kwargs)
        self.env_kwargs = env_kwargs
        self.gif_dir = Path(gif_dir)
        self.gif_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.prefix = prefix
        self.count = 0

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        # 当 eval 完成时，n_calls % eval_freq == 0
        if self.n_calls % self.eval_freq == 0:
            # 构造带 mask 的 demo 环境
            demo_env = ActionMasker(
                PathPlanningMaskEnv(**self.env_kwargs),
                lambda e: e.action_masks()
            )
            obs, _ = demo_env.reset()
            frames = []
            done = False
            while not done:
                #action, _ = self.model.predict(obs, deterministic=True)
                # 明确把 mask 传给 predict，让 MaskablePPO 在推断时也遵守掩码
                action, _ = self.model.predict(obs, deterministic=True, action_masks=obs["action_mask"])
                obs, _, done, _, _ = demo_env.step(action)
                # ← 直接调用底层 env 的 render(mode="rgb_array")
                frame = demo_env.env.render(mode="rgb_array")
                frames.append(frame)

            out = self.gif_dir / f"{self.prefix}_{self.count:03d}.gif"
            imageio.mimsave(out, frames, fps=self.fps)
            print(f"[EvalAndGif] saved {out}")
            self.count += 1

        return continue_training

def make_maskable_env(scen_file, problem_index, dynamics_config, footprint_config):
    base_env = PathPlanningMaskEnv(
        scen_file=scen_file,
        problem_index=problem_index,
        dynamics_config=dynamics_config,
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
        scen_file       = cfg["scen_file"],
        problem_index   = int(cfg.get("problem_index", 0)),
        dynamics_config = cfg["dynamics_config"],
        footprint_config= cfg["footprint_config"],
    )
    total_timesteps  = int(cfg.get("total_timesteps", 100_000))
    eval_freq        = int(cfg.get("eval_freq",     10_000))
    reward_threshold = float(cfg.get("reward_threshold",   1.0))

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

    # 4) 回调
    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
    eval_cb = EvalAndGifCallback(
        env_kwargs             = env_kwargs,
        gif_dir                = str(gif_dir),
        prefix                 = "eval",
        fps                    = 10,
        # 下面是传给 EvalCallback 的参数
        eval_env               = eval_env,
        best_model_save_path   = str(best_dir),
        callback_on_new_best   = stop_cb,
        eval_freq              = eval_freq,
        n_eval_episodes        = 1,
        deterministic          = True,
        render                 = False,
        verbose                = 1,
    )

    # 5) MaskablePPO
    model = MaskablePPO(
        policy            = "MultiInputPolicy",
        env               = train_env,
        learning_rate     = cfg.get("learning_rate", 3e-4),
        n_steps           = cfg.get("n_steps",    2048),
        batch_size        = cfg.get("batch_size",   64),
        n_epochs          = cfg.get("n_epochs",     10),
        gamma             = cfg.get("gamma",      0.99),
        clip_range        = cfg.get("clip_range",  0.2),
        ent_coef          = cfg.get("ent_coef",    0.0),
        tensorboard_log   = str(tb_dir),
        verbose           = 1,
    )

    # 6) 训练
    model.learn(total_timesteps=total_timesteps, callback=eval_cb)
    model.save(str(logdir / "final_model"))
    print("✅ Training complete. Outputs saved in", args.logdir)

if __name__ == "__main__":
    main()
