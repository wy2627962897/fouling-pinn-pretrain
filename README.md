# 基于 PINN 的换热器结垢热阻演化预测

Physics-Informed Neural Network for Heat Exchanger Fouling Resistance Prediction

## 项目概述

换热器结垢是化工行业最主要的能耗损失来源之一。本项目构建了从物理仿真 → 数据驱动建模 → 物理信息神经网络（PINN）的完整研究管线，系统论证了 PINN 在结垢预测数据稀缺场景下的核心优势。

**核心发现**：
- 稀疏数据（N < 500）下，PINN 的预测误差比纯 MLP 低 **100-200 倍**
- PINN 将物理单调性违反率从 MLP 的 **53.3% 降至 0.0%**
- 在数据充足场景下，MLP 和 PINN 均能达到 R² > 0.9

## 项目结构

```
├── 技术报告_基于PINN的换热器结垢预测.md   # 5-8 页技术报告
├── 笔记_换热器结垢基础.md                # 学习笔记
├── 文献综述_结垢预测与PINN.md            # 24 篇论文综述
├── README.md
├── concept-map*.mmd                      # 概念图 (Obsidian 可渲染)
├── code/
│   ├── fouling_simulator.py              # Kern-Seaton 物理仿真器
│   ├── visualize.py                      # 仿真可视化 (6 张图)
│   ├── data_generator.py                 # 合成数据生成器
│   ├── train_baseline.py                 # MLP 基线训练
│   ├── evaluate.py                       # 基线评估可视化
│   ├── pinn_model.py                     # PINN 模型定义 + 训练
│   ├── final_experiments.py              # PINN vs MLP 最终对比实验
│   └── output/
│       ├── fig*_*.png                    # 仿真图表
│       ├── eval_*.png                    # MLP 评估图表
│       ├── pinn_vs_mlp_sparse.png        # PINN vs MLP 对比
│       └── figures/
│           ├── fig1_sparse_advantage.png
│           ├── fig2_extrapolation.png
│           └── fig3_physical_consistency.png
```

## 快速开始

### 环境要求

```bash
conda create -n fouling-pinn python=3.12
conda activate fouling-pinn
conda install scikit-learn pandas matplotlib
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 运行仿真器

```bash
cd code
python fouling_simulator.py        # 基准工况 + 多工况对比
python visualize.py                # 生成 6 张仿真图表
```

### 生成数据 + 训练 MLP Baseline

```bash
python data_generator.py           # 生成 10,000 工况合成数据
python train_baseline.py           # MLP 训练 (三个实验)
python evaluate.py                 # 评估图表
```

### 训练 PINN + 对比实验

```bash
python final_experiments.py        # PINN vs MLP 三组实验 + 图表
```

## 实验设计

| 实验 | 场景 | MLP | PINN | 结论 |
|---|---|---|---|---|
| 稀疏数据 | N=20~2000 | R² 负值, 崩溃 | MAE 稳定 ~6e-4 | PINN 优势 100-200x |
| 外推 | 训≤3年 测>3年 | R²=0.70 | R²≈0 | MLP 赢 (PINN 需优化) |
| 物理一致性 | 单调性检查 | 53.3% 违反 | **0.0%** 违反 | PINN 完美 |

## 许可证

MIT
