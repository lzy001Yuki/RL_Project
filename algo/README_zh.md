# `algo/`：算法与显式规划模块

这个目录用于放置你自己的算法实现（区别于 `_instructor/` 的示例 baseline）。

目标：
- **显式规划（planning）**：把点云转成栅格，然后用 A* / 代价扰动生成多条“真实可行”的全局路线。
- **RL 低层控制（PPO / Off-policy）**：学习从当前位置到下一个子目标（landmark）的局部控制策略，负责平滑动作和避障。

> 说明：`algo/planning` 设计为 **torch-free**（仅依赖 numpy），方便你在没有 GPU/torch 的环境下先调通规划与数据管线。  
> `algo/ppo` 与 `algo/sac` 需要 PyTorch（训练时请安装 `torch`）。

---

## 1) 显式规划：`algo/planning`

### 从点云构建占据栅格（带安全膨胀）

```python
import numpy as np
from algo.planning import GridMap2D

points = np.load("data/pointcloud_2d.npy")           # (N,2)
grid = GridMap2D.from_pointcloud(
    points,
    resolution_m=2.0,
    padding_m=5.0,
    inflation_radius_m=4.0,  # 建议 > 2m，留出真实安全边界
)
```

### A* 规划一条路线，并抽取路标（landmarks）

```python
from algo.planning import astar, extract_landmarks_from_path

start_rc = grid.world_to_grid(x_start, y_start)
goal_rc  = grid.world_to_grid(x_goal,  y_goal)

result = astar(grid, start_rc, goal_rc)
path_xy = result.path_xy                       # 稠密路径点
landmarks = extract_landmarks_from_path(path_xy, max_landmarks=5)
```

### 生成多条“不同但合理”的路线（路径排斥 repulsion）

```python
from algo.planning import MultiPathConfig, plan_diverse_paths

cfg = MultiPathConfig(
    n_paths=80,
    repulsion_strength=2.0,
    repulsion_weight=2.0,
    detour_ratio_max=1.8,   # 控制绕行程度，保证真实
)
paths = plan_diverse_paths(grid, start_rc, goal_rc, cfg)  # List[AStarResult]
```

你可以把不同的 `paths[i]` 转成路标序列，交给 RL 低层执行，从而稳定地产出宏观多样的轨迹。

---

## 2) PPO：`algo/ppo`

已提供：
- `algo/ppo/policy.py`：`GaussianActorCritic`（连续动作，tanh squashing）
- `algo/ppo/storage.py`：`DictRolloutStorage`
- `algo/ppo/ppo.py`：PPO 更新逻辑

预期用法：写一个训练循环（类似 `_instructor/train.py`），把 env 产生的 dict obs 塞进 `DictRolloutStorage`，每 `num_steps` 更新一次 PPO。

仓库根目录提供了一个可直接运行的训练入口（基于规划采样子目标）：
- `train_ppo.py`

---

## 3) Off-policy SAC：`algo/sac`

已提供：
- `algo/sac/replay_buffer.py`：`DictReplayBuffer`
- `algo/sac/networks.py`：`SACActor` / `SACCritic`
- `algo/sac/sac.py`：SAC 更新逻辑（含自动温度 `alpha`）

预期用法：
1. 与环境交互，把 `(obs, action, reward, next_obs, done)` 放进 replay buffer
2. 每若干步从 buffer 采样 batch，调用 `SAC.update(batch)`

仓库根目录提供了一个可直接运行的训练入口（基于规划采样子目标）：
- `train_sac.py`

---

## 4) 生成提交：`submission/`

仓库根目录提供了一个提交轨迹生成脚本（显式规划 + 外部记忆 + 终点减弱）：
- `generate_submission.py`

最简单（规划-only，无需 torch）：

```bash
python generate_submission.py \
  --pointcloud_path data/pointcloud_2d.npy \
  --initials_path data/eval_initials_100.json \
  --output_dir submission
```

如果你想用训练好的低层控制器跟踪规划路标（需要 torch）：

```bash
python generate_submission.py \
  --pointcloud_path data/pointcloud_2d.npy \
  --initials_path data/eval_initials_100.json \
  --output_dir submission \
  --executor ppo_residual \
  --policy_path saved_data/ppo_xxx/controllers/final_policy.pt
```

说明：
- `--executor ppo_residual`：用于本仓库当前 `train_ppo.py` 默认训练的 Residual PPO（`action = guide(A*) + residual(PPO)`）
- `--executor ppo`：用于旧的“策略直接输出完整动作”的 PPO

## 下一步建议

1) 把环境改成“目标条件/子目标条件”：每个 step 的观测里包含 `subgoal_rel = (subgoal_xy - curr_xy)`  
2) 用 `plan_diverse_paths` 生成多条路线 → 抽 landmarks → 低层 RL 执行  
3) 每个 initial 先生成候选池（>20 条成功轨迹），再用 DTW 选集（DPP / greedy farthest）挑最分散的 20 条

---

## 依赖提示

- `algo/planning` 仅依赖 numpy（torch-free）。
- `algo/envs/uav_subgoal_env.py` 会优先使用 `scipy.spatial.cKDTree` 加速最近障碍查询；如果你的环境没有 SciPy，会自动退化为 brute force（会很慢）。

---

## 训练与多卡（推荐看这里）

仓库根目录的 `USAGE_zh.md` 提供了：
- PPO / SAC 的单卡训练命令
- `torchrun` 多卡/多机启动方式
- 如何用训练好的控制器配合 `generate_submission.py` 生成提交

入口脚本：
- `train_ppo.py`
- `train_sac.py`

---

## 5) 记忆条件 Goal-PPO（MC‑GC‑PPO）

如果你不想依赖“显式规划 → 路标 → 低层跟踪”，本仓库还提供了一个更偏 RL 的方案骨架：

- 外部记忆（per-initial repulsion memory）：`algo/memory/repulsion_memory.py`
- 记忆条件 Goal 环境包装：`algo/envs/uav_memory_goal_env.py`
- 训练 + 采样入口：仓库根目录 `train_mc_gc_ppo.py`

它支持两种常用工作流：
1) `--mode train --collect_during_train`：边训练边把成功轨迹按 baseline 格式落盘
2) `--mode collect`：加载策略，阶段 B 单独采集（每次成功后更新外部记忆，推动多样性）

命令示例见仓库根目录 `USAGE_zh.md`。
