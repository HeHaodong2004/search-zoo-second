#!/usr/bin/env python3
# debug_scan.py

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- 修改以下 import 路径为你项目中实际的 env 文件位置 ---
from domains_cc.path_planning_rl.env.planning_env import PathPlanningMaskEnv  
# 例如，如果你的 env 在 domains_cc/path_planning_rl/env/planning_env.py：
# from domains_cc.path_planning_rl.env.planning_env import PathPlanningMaskEnv

# --- 给环境打补丁，添加 render_scan_debug ---
def render_scan_debug(self, mode="human"):
    obs = self._get_obs()
    scans = obs["scan"]
    x, y, theta = self.state[:3]

    fig, ax = plt.subplots(figsize=(6,6))
    # 背景：地图和机器人
    addGridToPlot(self.world_cc, ax)      # 来自 worldCCVisualizer
    ax.plot(x, y, 'bo', label="robot")

    for idx, dist in enumerate(scans):
        ang = theta + self.laser_angles[idx]
        end_x = x + np.cos(ang) * dist
        end_y = y + np.sin(ang) * dist

        # 记下网格索引 row=y, col=x
        ix = int(np.floor(end_x / self.resolution))
        iy = int(np.floor(end_y / self.resolution))

        ax.plot([x, end_x], [y, end_y], 'r-', linewidth=0.5)
        ax.scatter(end_x, end_y, c='r', s=5)
        ax.text(end_x, end_y, f"({iy},{ix})", color='white', fontsize=6)

    ax.set_title(f"Scan Debug @ step {self._steps}")
    ax.set_aspect('equal')

    # 保存或显示
    os.makedirs("debug_scans", exist_ok=True)
    filename = f"debug_scans/step_{self._steps:03d}.png"
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"[ScanDebug] saved {filename}")

# patch into class
PathPlanningMaskEnv.render_scan_debug = render_scan_debug


# --- 如果需要 worldCCVisualizer 中的 addGridToPlot ---
try:
    from domains_cc.worldCCVisualizer import addGridToPlot
except ImportError:
    # 若路径不同，再调整这一行
    from worldCCVisualizer import addGridToPlot


def main():
    # 1. 环境参数：请根据实际路径填入 scen_file、dynamics_config、footprint_config
    scen_file       = "domains_cc/benchmark/moving-ai/scens/maze-32-32-2@basic.scen"
    dynamics_config = "domains_cc/dynamics/unicycle.yaml"
    footprint_config= "domains_cc/footprints/footprint_rectangle.yaml"


    env = PathPlanningMaskEnv(
        scen_file=scen_file,
        problem_index=0,
        dynamics_config=dynamics_config,
        footprint_config=footprint_config,
        max_steps=200,
        n_lasers=16,
    )

    obs, _ = env.reset()
    # 2. 随机或手动执行若干步
    for _ in range(50):
        # 随机选一个可行动作
        mask = env.action_masks()
        valid_actions = np.nonzero(mask)[0]
        action = np.random.choice(valid_actions)
        obs, reward, done, _, info = env.step(action)

        # 每一步都可视化一次
        env.render_scan_debug()

        if done:
            print("Episode done:", info)
            break

    print("Debug run complete. 查看 debug_scans/ 目录下的图片。")
'''
usage example
'''
if __name__ == "__main__":
    main()
