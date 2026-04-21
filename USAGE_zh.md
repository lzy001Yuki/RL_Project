# 代码使用说明（训练 + 多卡 + 生成提交）

这份说明面向本仓库当前实现的“**显式规划 + 子目标条件低层控制（PPO / SAC）+ 外部记忆多路径**”方案。

> 你要求“不要在这台机器上训练”：这里仅给出**如何在你自己的 GPU 机器/集群上训练**的命令与注意事项。

---

## 0) 你会用到的脚本

- 训练（基于我们的“规划采样子目标”方法）  
  - `train_ppo.py`：PPO 低层控制器（on-policy）  
  - `train_sac.py`：SAC 低层控制器（off-policy）

- 训练 + 采样（MC‑GC‑PPO：记忆条件 Goal-Conditioned PPO）  
  - `train_mc_gc_ppo.py`：支持 `--mode train`（可边训边存轨迹）与 `--mode collect`（阶段 B 单独采集）

- 生成提交（显式规划 + 外部记忆 + 终点减弱，可选用训练好的控制器执行）  
  - `generate_submission.py`

- 评估（官方）  
  - `tools/compute_diversity.py`  
  - `tools/evaluate_submission.py`

---

## 1) 环境与依赖

建议在你的 GPU 机器上创建独立环境：

```bash
conda create -n rl_assignment python=3.10 -y
conda activate rl_assignment
```

依赖（最低要求）：
- `numpy`
- `scipy`（强烈建议，用于 `cKDTree` 加速最近障碍查询；否则训练会非常慢）
- `torch`（建议 CUDA 版本与驱动匹配）

示例（按你的 CUDA 版本选择正确的 torch 安装方式）：

```bash
pip install numpy scipy
pip install torch
```

数据文件：
- `data/pointcloud_2d.npy`
- `data/eval_initials_100.json`

---

## 2) 单卡训练：PPO（推荐先跑通）

PPO 训练入口：`train_ppo.py`  
它会在每个 episode 前用 A* 规划出 start→goal 路径，然后从路径上**随机采样一个子目标 subgoal**，训练低层控制器去“到达 subgoal + 避障”。

当前 `train_ppo.py` 默认采用 **Residual RL**（更稳）：
- `action_env = action_guide(A*) + action_residual(PPO)`
- 其中 `action_guide` 是沿着 A* 路径的 lookahead 引导动作
- PPO 只学习一个幅度受限的残差动作用于修正/避障

```bash
python train_ppo.py \
  --pointcloud_path data/pointcloud_2d.npy \
  --initials_path data/eval_initials_100.json \
  --save_dir saved_data/ppo_subgoal
```

输出：
- 策略权重：`saved_data/ppo_subgoal/controllers/final_policy.pt`
- 日志：`saved_data/ppo_subgoal/train_log.txt`
- 配置：`saved_data/ppo_subgoal/config.json`

常用调参：
- `--num_steps`：每次 PPO 更新采样的步数（增大可提升样本效率，但更慢）
- `--max_iter`：迭代次数（总 env steps ≈ `max_iter * num_steps`）
- `--grid_inflation_radius_m`：栅格膨胀半径（越大越安全，但规划更容易“堵死”）
- Residual RL：
  - `--residual_frac`：PPO 残差占动作上限比例（默认 0.35）
  - `--guide_lookahead_m`：A* 引导 lookahead 距离（默认 10m）

---

## 3) 单卡训练：SAC（off-policy）

SAC 训练入口：`train_sac.py`

```bash
python train_sac.py \
  --pointcloud_path data/pointcloud_2d.npy \
  --initials_path data/eval_initials_100.json \
  --save_dir saved_data/sac_subgoal
```

输出：
- Actor 权重：`saved_data/sac_subgoal/controllers/final_actor.pt`
- Critic 权重：`saved_data/sac_subgoal/controllers/final_critic.pt`
- 日志：`saved_data/sac_subgoal/train_log.txt`

常用调参：
- `--start_steps`：前多少步用随机动作（探索更强，但收敛慢）
- `--update_after`：多少步后开始更新网络
- `--updates_per_step`：每个 env step 做多少次梯度更新

---

## 4) 多卡训练（DDP 风格，同步梯度平均）

本仓库的 PPO/SAC 更新逻辑在检测到 `torch.distributed` 初始化后，会做**同步梯度 all-reduce（平均）**；训练入口脚本支持 `torchrun` 启动。

关键点：
- **只在 rank0 保存模型/日志**（避免多进程写同一文件）
- **模型初始化在所有 rank 完全一致**，随后每个 rank 用不同随机种子采样数据（保证并行数据多样性）

### 4.1 单机多卡

4 卡示例（PPO）：

```bash
torchrun --standalone --nproc_per_node=4 train_ppo.py \
  --pointcloud_path data/pointcloud_2d.npy \
  --initials_path data/eval_initials_100.json \
  --save_dir saved_data/ppo_subgoal_ddp
```

4 卡示例（SAC）：

```bash
torchrun --standalone --nproc_per_node=4 train_sac.py \
  --pointcloud_path data/pointcloud_2d.npy \
  --initials_path data/eval_initials_100.json \
  --save_dir saved_data/sac_subgoal_ddp
```

### 4.2 多机多卡（集群）

两机示例（每机 8 卡）：

在 node0：

```bash
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  train_ppo.py --pointcloud_path data/pointcloud_2d.npy --initials_path data/eval_initials_100.json
```

在 node1：

```bash
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=8 \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  train_ppo.py --pointcloud_path data/pointcloud_2d.npy --initials_path data/eval_initials_100.json
```

> 多机时 `MASTER_ADDR/MASTER_PORT` 需要你按集群网络设置好。

### 4.3 单机多卡：MC‑GC‑PPO（记忆条件 Goal-PPO）

训练（只训练，不单独采样）：

```bash
torchrun --standalone --nproc_per_node=4 train_mc_gc_ppo.py \
  --mode train \
  --pointcloud_path data/pointcloud_2d.npy \
  --initials_path data/eval_initials_100.json \
  --save_dir saved_data/mc_gc_ppo_ddp
```

训练 + 边训边采集（示例：取前 20 个 initial，每个存 100 条成功轨迹；存满即停）：

```bash
torchrun --standalone --nproc_per_node=4 train_mc_gc_ppo.py \
  --mode train \
  --pointcloud_path data/pointcloud_2d.npy \
  --initials_path data/eval_initials_100.json \
  --save_dir saved_data/mc_gc_ppo_ddp_collect \
  --collect_during_train \
  --collect_take_first_n 20 \
  --collect_trajs_per_initial 100 \
  --stop_when_collected
```

输出（默认）：
- 策略：`saved_data/mc_gc_ppo_ddp_collect/controllers/final_policy.pt`
- 轨迹：`saved_data/mc_gc_ppo_ddp_collect/baseline_trajs/initial_*/traj_*.txt`

---

## 5) 生成提交轨迹（规划-only / 使用控制器执行）

### 5.1 规划-only（无需 torch，最稳）

```bash
python generate_submission.py \
  --pointcloud_path data/pointcloud_2d.npy \
  --initials_path data/eval_initials_100.json \
  --output_dir submission
```

### 5.2 使用 PPO 低层执行（需要 torch）

如果你的 PPO 是 **Residual 模式**（本仓库当前 `train_ppo.py` 默认训练的就是这种），用：

```bash
python generate_submission.py \
  --pointcloud_path data/pointcloud_2d.npy \
  --initials_path data/eval_initials_100.json \
  --output_dir submission \
  --executor ppo_residual \
  --policy_path saved_data/ppo_subgoal/controllers/final_policy.pt
```

如果你使用的是旧的“PPO 直接输出完整动作”的模型，用 `--executor ppo`。

### 5.3 使用 SAC 低层执行（需要 torch）

```bash
python generate_submission.py \
  --pointcloud_path data/pointcloud_2d.npy \
  --initials_path data/eval_initials_100.json \
  --output_dir submission \
  --executor sac \
  --policy_path saved_data/sac_subgoal/controllers/final_actor.pt
```

### 5.4 MC‑GC‑PPO：阶段 B 单独采集（不依赖 `generate_submission.py`）

这个模式会直接 rollout 你的策略，并在每次成功后更新“外部记忆”（repulsion memory），从而让同一个 initial 逐渐走出不同路线。

```bash
torchrun --standalone --nproc_per_node=4 train_mc_gc_ppo.py \
  --mode collect \
  --pointcloud_path data/pointcloud_2d.npy \
  --initials_path data/eval_initials_100.json \
  --take_first_n 20 \
  --trajs_per_initial 100 \
  --output_dir submission/mc_gc_ppo \
  --policy_path saved_data/mc_gc_ppo_ddp/controllers/final_policy.pt
```

常见调参：
- 采集失败/成功率低：调小 `--memory_reward_weight` 或调大 `--fade_w_min` / `--fade_near_m`
- 多样性不够：调大 `--memory_strength_success` / `--memory_radius_cells`，并保持 `--deterministic` 关闭（默认随机动作）

---

## 6) 自测评估（建议生成后立刻跑）

```bash
python tools/evaluate_submission.py \
  --submission_dir submission \
  --initials_path data/eval_initials_100.json \
  --baseline_path data/baseline_diversity.json
```

仅计算多样性：

```bash
python tools/compute_diversity.py \
  --trajs_dir submission \
  --initials_path data/eval_initials_100.json
```

---

## 7) 计算资源预估（经验级）

这个项目的网络很小（MLP，obs=6，act=2），**显存占用很低**；训练速度往往更受 **CPU 环境步进（最近障碍查询）**影响。

一个保守的预估方式：
- PPO：总步数约 `max_iter * num_steps`（默认 2e4 * 256 ≈ 5.1M env steps / rank）
- SAC：总步数约 `max_env_steps`（默认 2.0M env steps / rank）

建议你先用小规模配置跑通：
- PPO：`--max_iter 2000 --num_steps 128`
- SAC：`--max_env_steps 200000 --start_steps 20000`

确认 reward 曲线在上升后，再上大规模训练与多卡。

---

## 8) 常见报错与排查

### 8.1 `RuntimeError: No valid initials could be planned on the current grid`

含义：训练脚本里的 `PlannedSubgoalSampler` 会用 A* 在栅格上规划 start→goal 的全局路径，并从中采样子目标。如果栅格构建/膨胀导致 **所有** 起点-终点都无法规划，就会报这个错。

常见原因与解决（按优先级）：
- 目标中心在建筑物内：A* 规划到“目标中心”会失败。现在代码会自动在 `--goal_success_radius_m` 范围内寻找**可达的自由格子**作为 proxy goal（默认 30m）。如果你用的是旧代码，请更新到最新版本。
- 栅格膨胀过大：把 `--grid_inflation_radius_m` 调小（例如 `4 -> 2` 或 `0`）验证能否规划成功。
- 栅格边界太紧：把 `--grid_padding_m` 调大（例如 `5 -> 30/50`），给绕行留出空间。
- 起点落在膨胀障碍里：把 `--start_snap_radius_m` 调大（例如 `10 -> 20`），允许起点“吸附”到附近自由格子。

示例（SAC，多卡也同理）：

```bash
python train_sac.py \
  --pointcloud_path data/pointcloud_2d.npy \
  --initials_path data/eval_initials_100.json \
  --grid_padding_m 50 \
  --grid_inflation_radius_m 2 \
  --goal_success_radius_m 30 \
  --start_snap_radius_m 20
```



• 按你现在这版代码，直接跑 train_ppo.py 就行。核心是：默认不启用 action guide，走“多路径
  + 拐点池 + 去重聚类 + 记忆惩罚 + PPO”。--use_action_guide 是可选开关
  （train_ppo.py:67）。

  1) 最小可运行（先验证流程）

  python train_ppo.py \
    --pointcloud_path data/pointcloud_2d.npy \
    --initials_path data/eval_initials_100.json \
    --save_dir saved_data/ppo_landmark_smoke \
    --max_iter 20 \
    --num_steps 64

  2) 正式训练（论文方法，推荐）

  python train_ppo.py \
    --pointcloud_path data/pointcloud_2d.npy \
    --initials_path data/eval_initials_100.json \
    --save_dir saved_data/ppo_landmark_full \
    --max_iter 20000 \
    --num_steps 256 \
    --n_paths 8 \
    --path_shape_top_k 5 \
    --landmark_turn_thresh_deg 25 \
    --landmark_dedup_radius_m 18 \
    --landmark_cluster_radius_m 25 \
    --landmark_max_per_initial 24 \
    --landmark_max_hops 12 \
    --memory_reward_weight 2.0

  3) 如果你要“旧版 residual+guide”一起用
  加 --use_action_guide（并可调 --residual_frac）：

  ... --use_action_guide --residual_frac 0.75

  4) 边训练边落盘轨迹（提交格式）

  python train_ppo.py \
    --pointcloud_path data/pointcloud_2d.npy \
    --initials_path data/eval_initials_100.json \
    --save_dir saved_data/ppo_landmark_collect \
    --collect_during_train \
    --collect_take_first_n 20 \
    --collect_trajs_per_initial 100 \
    --stop_when_collected

  输出轨迹在 saved_data/ppo_landmark_collect/baseline_trajs（train_ppo.py:157 附近参数定
  义）。

  5) 只看规划结果（不训练）

  python train_ppo.py \
    --pointcloud_path data/pointcloud_2d.npy \
    --initials_path data/eval_initials_100.json \
    --save_dir saved_data/plan_only \
    --plan_only \
    --n_paths 8 \
    --path_shape_top_k 5

  如果你愿意，我可以再给你一组“更稳成功率”与“更高多样性”两套参数预设。