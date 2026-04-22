# RL Assignment指南

[paper](paper.pdf) [trajectories collection](submission/baseline_trajs/)

---

## 1. 环境准备

```bash
conda create -n rl_assignment python=3.10 -y
conda activate rl_assignment
pip install numpy scipy torch matplotlib
```

数据文件默认使用仓库内：
- `data/pointcloud_2d.npy`
- `data/eval_initials_100.json`
- `data/baseline_diversity.json`

---

## 2. 训练

当前论文方法对应脚本是：
- `train_ppo.py`

其核心流程是：
1. 动态代价多路径 A*（multi-path）
2. 拐点池 + 去重 + 聚类（landmark pool）
3. 基于记忆惩罚的 landmark 选择
4. 目标条件 PPO 低层控制


Baseline 为
- `_instructor/` 里的简单 PPO 实现




```bash
python train_ppo.py \
  --pointcloud_path data/pointcloud_2d.npy \
  --initials_path data/eval_initials_100.json \
  --save_dir saved_data/ppo_paper_method \
  --max_iter 100000 \
  --num_steps 256 \
  --n_paths 8 \
  --path_shape_top_k 5 \
  --landmark_turn_thresh_deg 25 \
  --landmark_dedup_radius_m 18 \
  --landmark_cluster_radius_m 25 \
  --landmark_max_per_initial 24 \
  --landmark_max_hops 12 \
  --memory_reward_weight 2.0
```
---

## 3. 训练同时采集提交轨迹

```bash
python train_ppo.py \
  --pointcloud_path data/pointcloud_2d.npy \
  --initials_path data/eval_initials_20.json \
  --save_dir saved_data/ppo_collect \
  --collect_during_train \
  --collect_take_first_n 20 \
  --collect_trajs_per_initial 100 \
  --stop_when_collected
```

输出目录：
- 轨迹：`saved_data/ppo_collect/baseline_trajs/initial_*/traj_*.txt`
- 规划轨迹：`saved_data/ppo_collect/plan/baseline_trajs/...`
- 日志：`saved_data/ppo_collect/train_log.txt`

## 4.评估
见[README_zh.md](README_zh.md)
