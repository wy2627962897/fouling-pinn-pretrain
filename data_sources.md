# 公开曲线数据源台账

> 目标：优先收集可数字化的换热器结垢曲线，用于真实/文献数据验证、多机制 ODE 拟合和 PINN 约束实验。

## 第一批种子论文

| 优先级 | source_id | 论文 | 数据类型 | 可提取图表 | 适合模型 | 备注 |
|---|---|---|---|---|---|---|
| P0 | source_001 | Jradi, Fguiri, Marvillet & Jeday, *Tubular Heat Exchanger Fouling in Phosphoric Acid Concentration Process* | 磷酸浓缩过程管式换热器工业数据 | Fig. 5: `Rf(t)`；Fig. 6: `U(t)`；Fig. 7: 实验值 vs Kern-Seaton 拟合 | Kern-Seaton 渐近模型；清洗后无诱导期模型 | 开放 PDF，曲线直接可数字化；文中给出 `Rf* = 1.72e-4 m2 K W-1`、`tau = 40.32 h`、`R2 = 0.975` |
| P0 | source_002 | Benyahia et al., *Study of the fouling deposit in the heat exchangers of Algiers refinery* | 炼油厂原油预热器现场数据 | Fig. 5: 两组换热器 `Rf(t)`；Fig. 9: deposit thickness vs time；Fig. 10: thickness vs `Rf` | Kern-Seaton 渐近模型；清洗/停机扰动模型；厚度-热阻耦合模型 | Springer open access；包含 E101 CBA 和 E101 FED 两组曲线，适合做跨设备验证 |
| P0 | source_003 | Riihimäki et al., *Crystallization Fouling on Modified Heat Transfer Surfaces* | CaCO3 结晶结垢实验数据 | Fig. 4: 未改性不锈钢重复 `Rf(t)`；Fig. 5: 涂层表面 `Rf(t)`；Fig. 11: 抛光/打磨表面 `Rf(t)`；Fig. 13: 图案表面 `Rf(t)` | 诱导期 + 线性增长模型；表面改性诱导期模型；单调/非单调早期约束 | 开放会议论文；曲线多，适合测试 induction period 和负 fouling resistance 早期现象 |

## 数字化顺序

1. 先做 `source_001`：目标是 3 条曲线，分别来自 Fig. 5、Fig. 6、Fig. 7。
2. 再做 `source_002`：目标是 4 条曲线，Fig. 5 两条 `Rf(t)`，Fig. 9 两条 thickness 曲线。
3. 最后做 `source_003`：目标是至少 8 条曲线，优先 Fig. 4 和 Fig. 13，因为曲线机制最清楚。

## 统一元数据字段

```csv
source_id,curve_id,paper_title,doi_or_url,fouling_type,system,figure,x_name,x_unit,y_name,y_unit,conditions,notes
```

## 曲线 CSV 字段

```csv
time,value
```

## 初始模型标签

| 标签 | 判断标准 |
|---|---|
| `asymptotic` | `Rf(t)` 快速增长后趋于平台 |
| `induction_linear` | 早期有诱导期，之后近似线性增长 |
| `negative_then_growth` | 初期 `Rf` 下降或为负，之后增长 |
| `cleaning_disturbed` | 曲线中途因清洗/停机下降 |
| `thickness_rf_coupled` | 同时有厚度与热阻，可拟合 `Rf = e_f / lambda_f` |

