# domains_cc/path_planning_rl/train/train_ppo.py
# -*- coding: utf-8 -*-
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
    每隔 eval_freq 步：
      1) 在 eval_env（VecMonitor(DummyVecEnv)）上跑 n_eval_episodes 次评估，
         统计平均 reward、平均步长、成功率；
      2) 用一个纯粹的 Maskable 单环境跑一条确定性轨迹，保存成 GIF。
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
        self.n_eval_episodes = eval_kwargs.get("n_eval_episodes", 5)

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        # 只有在真正执行完一次完整 eval 后才触发 GIF+success 统计
        if self.n_calls % self.eval_freq == 0:
            # —— 1) 在向量化环境上评估 —— 
            rewards, lengths, successes = [], [], 0
            for _ in range(self.n_eval_episodes):
                # VecMonitor(DummyVecEnv) 的 reset 仅返回 obs（不返回 info）
                obs = self.eval_env.reset()
                done = np.zeros((self.eval_env.num_envs,), dtype=bool)
                total_r, steps, last_info = 0.0, 0, {}
                while not done.all():
                    action, _ = self.model.predict(
                        obs,
                        deterministic=True,
                        action_masks=obs["action_mask"]
                    )
                    # 这里 VecEnv.step 只返回 4 个值
                    obs, rews, dones, infos = self.eval_env.step(action)
                    done = dones
                    total_r += rews[0]
                    steps += 1
                    last_info = infos[0]
                rewards.append(total_r)
                lengths.append(steps)
                if last_info.get("reached", False):
                    successes += 1

            mean_r = np.mean(rewards)
            std_r = np.std(rewards)
            mean_l = np.mean(lengths)
            succ_rate = successes / self.n_eval_episodes
            print(f"[Eval] episodes={self.n_eval_episodes}  "
                  f"mean_reward={mean_r:.2f}±{std_r:.2f}  "
                  f"mean_length={mean_l:.1f}  "
                  f"success_rate={succ_rate:.0%}")

            # —— 2) 在纯单环境上跑一条 deterministic rollout 生成 GIF —— 
            demo_env = ActionMasker(
                PathPlanningMaskEnv(**self.env_kwargs),
                lambda e: e.action_masks()
            )
            reset_out = demo_env.reset()
            if isinstance(reset_out, tuple):
                obs, _ = reset_out
            else:
                obs = reset_out
            frames = []
            done = False
            info = {}
            while not done:
                action, _ = self.model.predict(
                    obs,
                    deterministic=True,
                    action_masks=obs["action_mask"]
                )
                step_out = demo_env.step(action)
                # 这里可能返回 4 或 5 个值，兼容两者
                if len(step_out) == 5:
                    obs, _, terminated, truncated, info = step_out
                    done = terminated or truncated
                else:
                    obs, _, done, info = step_out
                frames.append(demo_env.env.render(mode="rgb_array"))

            gif_path = self.gif_dir / f"{self.prefix}_{self.count:03d}.gif"
            imageio.mimsave(str(gif_path), frames, fps=self.fps)
            print(f"[EvalAndGif] saved {gif_path}")
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
        scen_file        = cfg["scen_file"],
        problem_index    = int(cfg.get("problem_index", 0)),
        dynamics_config  = cfg["dynamics_config"],
        footprint_config = cfg["footprint_config"],
    )
    total_timesteps  = int(cfg.get("total_timesteps", 100_000))
    eval_freq        = int(cfg.get("eval_freq",     10_000))
    reward_threshold = float(cfg.get("reward_threshold", 10.0))

    # 2) 创建输出目录
    logdir   = Path(args.logdir)
    tb_dir   = logdir / "tensorboard"
    best_dir = logdir / "best_model"
    gif_dir  = logdir / "gifs"
    for d in (logdir, tb_dir, best_dir, gif_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 3) 向量化 + MaskableWrapper
    def _make(): return make_maskable_env(**env_kwargs)
    train_env = VecMonitor(DummyVecEnv([_make]))
    eval_env  = VecMonitor(DummyVecEnv([_make]))

    # 4) 回调设置
    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
    eval_cb = EvalAndGifCallback(
        env_kwargs            = env_kwargs,
        gif_dir               = str(gif_dir),
        prefix                = "eval",
        fps                   = 10,
        # 下面参数传给父类 EvalCallback
        eval_env              = eval_env,
        best_model_save_path  = str(best_dir),
        callback_on_new_best  = stop_cb,
        eval_freq             = eval_freq,
        n_eval_episodes       = cfg.get("n_eval_episodes", 5),
        deterministic         = True,
        render                = False,
        verbose               = 1,
    )

    # 5) MaskablePPO 主训练循环
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

    # 6) 开始训练
    model.learn(total_timesteps=total_timesteps, callback=eval_cb)
    model.save(str(logdir / "final_model"))
    print("✅ Training complete. Outputs saved in", args.logdir)


if __name__ == "__main__":
    main()
