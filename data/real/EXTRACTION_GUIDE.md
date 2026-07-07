# 论文图表曲线提取操作指南

## 工具

推荐先用 WebPlotDigitizer：

https://automeris.io/WebPlotDigitizer/

原因：人工校准坐标轴更可靠，适合论文图像中坐标轴、图例、网格线不完全标准的情况。

## 已生成的页图

目标图页已从 PDF 渲染为 PNG：

```text
data/real/sources/source_001_page_12-12.png
data/real/sources/source_001_page_13-13.png
data/real/sources/source_001_page_14-14.png
data/real/sources/source_002_page_05-5.png
data/real/sources/source_002_page_06-6.png
data/real/sources/source_002_page_07-7.png
data/real/sources/source_003_page_03-3.png
data/real/sources/source_003_page_04-4.png
data/real/sources/source_003_page_05-5.png
data/real/sources/source_003_page_06-6.png
```

## 第一条曲线：source_001 Fig. 5

目标：提取磷酸浓缩管式换热器的 `Rf(t)`。

1. 打开 WebPlotDigitizer。
2. 点击 `Load Image`，选择：

```text
data/real/sources/source_001_page_12-12.png
```

3. 选择 `2D (X-Y) Plot`。
4. 坐标轴校准：
   - X 轴选择图中两个已知点，例如 `0 h` 和最大时间点。
   - Y 轴选择图中两个已知点，例如 `0` 和图中最大 `Rf` 刻度。
   - 具体刻度以图片上坐标轴文字为准，不要用肉眼估计单位。
5. 使用 `Manual Extraction` 点曲线上的实验点。
6. 如果图中有散点和拟合线，先只提取实验散点；拟合线可以另存一条曲线。
7. 导出 CSV，保存为：

```text
data/real/curves/source_001_fig5_rf_exp.csv
```

CSV 格式保持：

```csv
time,value
0,0
...
```

8. 在 `metadata.csv` 中确认 `source_001_fig5_rf` 对应的信息完整。

## 推荐提取顺序

1. `source_001_fig5_rf_exp.csv`：第一条真实 `Rf(t)` 曲线。
2. `source_001_fig7_rf_exp.csv`：实验值。
3. `source_001_fig7_rf_ks_fit.csv`：Kern-Seaton 拟合线。
4. `source_002_fig5_rf_cba.csv`：炼油厂 E101 CBA。
5. `source_002_fig5_rf_fed.csv`：炼油厂 E101 FED。
6. `source_003_fig4_unmodified_run1.csv`：CaCO3 未改性表面一条曲线。
7. `source_003_fig13_flat.csv` 和 patterned curves：诱导期/线性增长模型。

## 质量控制

每条曲线提取后检查：

- 起点是否符合论文描述。
- 单位是否与论文一致。
- 曲线趋势是否与图中一致。
- 是否误把拟合线当作实验点。
- 多曲线图中颜色/符号是否对应正确图例。
- 如果坐标轴是科学计数法，导出前统一换算到 SI 或论文原单位。

## 命名规范

```text
source_{编号}_fig{图号}_{变量}_{曲线说明}.csv
```

示例：

```text
source_001_fig5_rf_exp.csv
source_001_fig7_rf_ks_fit.csv
source_002_fig5_rf_cba.csv
source_003_fig13_rf_flat.csv
```

