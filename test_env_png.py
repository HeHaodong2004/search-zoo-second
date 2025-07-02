# domains_cc/path_planning_rl/test/test_motion_primitives_vis.py

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from domains_cc.worldCCVisualizer import addGridToPlot
from domains_cc.path_planning_rl.env.planning_env import PathPlanningMaskEnv

@pytest.fixture(scope="module")
def env():
    return PathPlanningMaskEnv(
        scen_file="domains_cc/benchmark/in2d/scens/Cantwell@basic.scen",
        problem_index=0,
        dynamics_config="domains_cc/dynamics/ackermann.yaml",
        footprint_config="domains_cc/footprints/footprint_rectangle.yaml"
    )

def test_visualize_primitives(tmp_path: Path, env):
    """
    画出每个 motion primitive 在当前 state 下的轨迹，
    将它们拼在一张大图里，保存为 PNG，方便快速检查。
    """
    # 1) 重置，拿到起始状态
    _ = env.reset()
    state = env.state.copy()

    # 2) 取出所有 primitive 的完整轨迹 (N, steps+1, state_dim)
    trajs = env.dynamics.get_next_states(state)[:, :, :3]  # 只要 x,y,theta

    # 3) 新建画布：N 列 1 行
    N = env.n_actions
    fig, axes = plt.subplots(
        nrows=1, ncols=N,
        figsize=(3*N, 3),
        squeeze=False
    )

    # 4) 对每条 primitive：
    for i in range(N):
        ax = axes[0, i]
        ax.set_title(f"P{i}")
        # 网格
        addGridToPlot(env.world_cc, ax)
        # 轨迹
        path = trajs[i]
        ax.plot(path[:, 0], path[:, 1], '-o', markersize=2)
        # 起点箭头
        ax.arrow(
            path[0,0], path[0,1],
            0.2*np.cos(path[0,2]), 0.2*np.sin(path[0,2]),
            head_width=0.1, head_length=0.1, fc='green', ec='green'
        )
        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    #out = tmp_path / "motion_primitives_all.png"
    #fig.savefig(out, dpi=150)
    out = Path("results") / "motion_primitives_all.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)

    plt.close(fig)

    assert out.exists(), f"可视化图应该生成在 {out}"
    print(f"[VIS] motion primitives visualization saved to {out}") 
