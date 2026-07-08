# 换热器结垢热阻演变——相关文献综述

> 根据概念图主题（结垢动力学 dRf/dt = φd − φr、Rf(t) 演化预测、清洗周期优化）按相关性和重要性排列。来源于 Semantic Scholar 和 OpenAlex。

---

## 一、基础理论与经典模型（必读）

### 1. Kern & Seaton (1959) — 奠基性模型
"A Theoretical Analysis of Thermal Surface Fouling"
*British Chemical Engineering, 4(5), 258-262*

**被引次数：** 数千次（经典论文）
**核心贡献：** 最早提出结垢的沉积-剥离竞争模型 dRf/dt = φd − φr。认为污垢热阻随时间呈渐近线型增长（asymptotic model），是概念图中动力学方程的理论源头。

### 2. Epstein (1983) — 全面分类框架
"Thinking about Heat Transfer Fouling: A 5 × 5 Matrix"
*Heat Transfer Engineering, 4(1), 43-56*

**核心贡献：** 提出结垢的 5×5 矩阵分类法（5种结垢机理 × 5个关键阶段），将结垢分为结晶、颗粒、化学反应、腐蚀、生物五类。结晶/化学反应结垢研究可直接引用此框架。

### 3. Ebert & Panchal (1995/1997) — 阈值模型
"Analysis of Exxon Crude-Oil-Slip Stream Coking Data"
*Fouling Mitigation of Industrial Heat-Exchange Equipment*

**核心贡献：** 提出**"阈值结垢"（threshold fouling）**概念——存在一个临界条件（温度、流速），低于此条件结垢不会发生。突破了过去只关注沉积-剥离平衡的框架。
**发展综述：** Wilson, Ishiyama & Polley (2017) — "Twenty Years of Ebert and Panchal—What Next?"

---

## 二、结垢动力学与动态建模（与概念图最相关）

### 4. Diaz-Bejarano, Coletti & Macchietto (2015) ⭐⭐⭐
"A New Dynamic Model of Crude Oil Fouling Deposits and Its Application to the Simulation of Fouling-Cleaning Cycles"
*AIChE Journal, 61(1), 233-250.* DOI: 10.1002/aic.15036

**被引：** 40次
**核心贡献：** 建立分布参数一维动态模型，不仅描述污垢热阻的变化，还考虑污垢层厚度、热导率老化（aging）、组分变化的时空演化。可用于模拟整个结垢-清洗循环周期。

### 5. Diaz-Bejarano, Coletti & Macchietto (2016a) ⭐⭐⭐
"Thermo-Hydraulic Analysis of Refinery Heat Exchangers Undergoing Fouling"
*AIChE Journal, 63(2), 662-676.* DOI: 10.1002/aic.15457

**被引：** 34次
**核心贡献：** 提出一套完整的热工水力动态分析系统方法，能从工厂数据中提取结垢状态、估算沉积物属性、评估结垢对换热器性能的影响。

### 6. Diaz-Bejarano, Coletti & Macchietto (2016b) ⭐⭐
"Crude Oil Fouling Deposition, Suppression, Removal, and Consolidation — and How to Tell the Difference"
*Heat Transfer Engineering, 37(13), 1095-1107.* DOI: 10.1080/01457632.2016.1206408

**被引：** 16次
**核心贡献：** 深入分析沉积、抑制、剥离、固结四种机制的差异，提出实验方法来区分这些效应。直接对应概念图中的 φd 和 φr 项。

### 7. Ishiyama, Paterson & Wilson (2013) ⭐⭐
"Aging is Important: Closing the Fouling–Cleaning Loop"
*Heat Transfer Engineering, 34(14), 1131-1142.* DOI: 10.1080/01457632.2013.825192

**被引：** 20次
**核心贡献：** 强调**污垢老化（aging）**对清洗-再结垢循环的影响——上一次清洗的效果决定了下次结垢的初始阶段，形成闭环。对清洗优化决策有重要参考。

### 8. Schluter, Augustin & Scholl (2020)
"Introducing a Holistic Approach to Model and Link Fouling Resistances"
*Heat and Mass Transfer, 57, 317-329.* DOI: 10.1007/s00231-020-03011-8

**被引：** 8次
**核心贡献：** 提出了将污垢热阻与过程参数全局关联的建模方法，超越了传统的单换热器模型。

---

## 三、结晶结垢（与温度/浓度驱动直接相关）

### 9. Loge, Anabaraonye & Fosbol (2022) ⭐⭐
"Growth Mechanisms of Composite Fouling: The Impact of Substrates on Detachment Processes"
*Chemical Engineering Journal, 446, 137008.* DOI: 10.1016/j.cej.2022.137008

**被引：** 20次
**核心贡献：** 研究了两种结晶污垢（BaSO₄ 和 CaCO₃）的复合沉积-剥离机理，首次揭示了基材对剥离过程的影响。与 φr（剥离项）研究直接相关。

### 10. Maddahi, Hatamipour & Jamialahmadi (2019) ⭐
"A Model for the Prediction of Thermal Resistance of Calcium Sulfate Crystallization Fouling in a Liquid–Solid Fluidized Bed Heat Exchanger"
*International Journal of Thermal Sciences, 145, 106034.* DOI: 10.1016/j.ijthermalsci.2019.106034

**被引：** 13次
**核心贡献：** 专门针对 CaSO₄ 结晶结垢建立了预测模型，给出关联式，直接关联温度、浓度与 φd（沉积项）。

### 11. Berce, Arhar, Hadzic & Zupancic et al. (2024) ⭐
"Boiling-Induced Surface Aging and Crystallization Fouling of Functionalized Smooth and Laser-Textured Copper Interfaces"
*Applied Thermal Engineering, 244, 122540.* DOI: 10.1016/j.applthermaleng.2024.122540

**被引：** 30次
**核心贡献：** 长期实验研究表面改性和沸腾条件对 CaSO₄ 结晶结垢的影响，研究了几百小时的结垢演变过程。

---

## 四、清洗周期优化（与优化目标直接相关）

### 12. Al Ismaili, Lee & Wilson et al. (2019) ⭐⭐⭐
"Optimisation of Heat Exchanger Network Cleaning Schedules: Incorporating Uncertainty in Fouling and Cleaning Model Parameters"
*Computers & Chemical Engineering, 121, 409-425.* DOI: 10.1016/j.compchemeng.2018.11.002

**被引：** 25次
**核心贡献：** 将参数不确定性纳入换热器网络清洗调度优化，证明确定性假设与考虑不确定性之间存在巨大的财务绩效差异。将清洗调度问题表述为多阶段混合整数最优控制问题（MIOCP）。

### 13. Lozano Santamaria & Macchietto (2020) ⭐⭐
"Online Integration of Optimal Cleaning Scheduling and Control of Heat Exchanger Networks under Fouling"
*Industrial & Engineering Chemistry Research, 59(13), 6131-6146.* DOI: 10.1021/acs.iecr.9b05905

**被引：** 12次
**核心贡献：** 首次提出在线集成清洗调度与流量控制的方法论，将"通过调节流量的控制动作"和"周期性清洗的调度动作"统一优化。与 min{能耗损失(t) + 清洗费用(t)} 框架一致。

### 14. De Cesaro, Ravagnani & Mele et al. (2024) ⭐
"Cleaning Schedule Optimization of Heat Exchangers with Fouling on Tube and Shell Sides: A Metaheuristic Approach"
*Energies, 17(1), 118.* DOI: 10.3390/en17010118

**被引：** 1次
**核心贡献：** 同时考虑管侧和壳侧结垢的清洗调度优化（以往研究常忽略壳侧），使用元启发式方法降低计算时间。

### 15. Mei, Kiyomoto & Kato et al. (2023) ⭐
"Data-Driven Soft Sensor for Crude Oil Fouling Monitoring in Heat Exchanger Networks"
*IEEE Sensors Journal, 23(15), 17352-17363.* DOI: 10.1109/JSEN.2023.3288927

**被引：** 4次
**核心贡献：** 建立数据驱动的软测量模型实时监测换热器网络的结垢状态，无需依赖复杂的机理模型即可推算 Rf(t)。其姊妹篇 (2022) 将该方法用于清洗调度优化。

---

## 五、基于机器学习/人工智能的结垢预测（前沿方法）

### 16. Jradi, Marvillet & Jeday (2022) ⭐⭐
"Analysis and Estimation of Cross-Flow Heat Exchanger Fouling in Phosphoric Acid Concentration Plant Using RSM and ANN"
*Scientific Reports, 12, 12353.* DOI: 10.1038/s41598-022-16638-0

**被引：** 18次
**核心贡献：** 对比了响应面法（RSM）和人工神经网络（ANN）预测磷酸浓缩换热器结垢热阻的效果。结垢机理为结晶结垢，与研究方向高度相关。

### 17. Jradi, Marvillet & Jeday (2023) ⭐
"Estimation and Sensitivity Analysis of Fouling Resistance in Phosphoric Acid/Steam Heat Exchanger Using ANN and Regression Methods"
*Scientific Reports, 13, 9148.* DOI: 10.1038/s41598-023-36078-4

**被引：** 12次
**核心贡献：** 进行敏感性分析，确定哪些操作参数（温度、流速、浓度）对 Rf 的影响最大。直接对应概念图中"壁温↑ → 结晶↑ → φd↑"的因果链。

### 18. Ikram, Djilali & Abdennasser et al. (2023) ⭐
"Comparative Analysis of Fouling Resistance Prediction in Shell and Tube Heat Exchangers Using Advanced ML Techniques"
*Research on Engineering Structures and Materials, 9(3), 1053-1073*

**被引：** 9次
**核心贡献：** 系统对比了三种AI技术（FNN-MLP, NARX, LSTM）对管壳式换热器结垢热阻的预测效果，结论是FFNN-MLP模型精度最高。

### 19. Liang, Zhu & Wang et al. (2024) ⭐
"Fouling Prediction of a Heat Exchanger Based on Wavelet Neural Network Optimized by Improved Particle Swarm Optimization"
*Processes, 12(4), 760.* DOI: 10.3390/pr12040760

**被引：** 8次
**核心贡献：** 提出改进粒子群算法优化的小波神经网络（IPSO-WNN）预测换热器结垢，在实验平台数据上验证了优于传统WNN和PSO-WNN。

### 20. Chen, Meng & Yu et al. (2024)
"A Prediction Model for Heat Exchanger Fouling Factor Based on Stacking Model"
*IEEE Latin America Transactions, 22(5), 414-421*

**被引：** 1次
**核心贡献：** 使用Stacking集成学习融合多个基模型预测污垢因子，应对石化行业节能降耗需求。

### 21. Ardsomang, Hines & Upadhyaya (2021)
"Heat Exchanger Fouling and Estimation of Remaining Useful Life"
*Annual Conference of the PHM Society, 5(1).* DOI: 10.36001/phmconf.2013.v5i1.2773

**被引：** 24次
**核心贡献：** 数据驱动方法估算换热器剩余使用寿命（RUL），直接对应概念图中的"清洗阈值"标注。

---

## 六、监测与诊断方法

### 22. Diaz-Bejarano, Coletti & Macchietto (2020) ⭐⭐
"A Model-Based Method for Visualization, Monitoring, and Diagnosis of Fouling in Heat Exchangers"
*Industrial & Engineering Chemistry Research, 59(17), 8253-8267.* DOI: 10.1021/acs.iecr.9b05490

**被引：** 22次
**核心贡献：** 提出**TH-λ图（温度-热导率图）**的可视化方法，能精确估计结垢位置、程度和沉积物属性。对 Rf(t) 曲线绘制有启发。

### 23. Zitouni, Fguiri & Assadi et al. (2025)
"Improving Heat Exchanger Fouling Detection for Phosphoric Acid Concentration Units: A Hybrid Inverse Approach Integrating GA and Levenberg-Marquardt"
*Case Studies in Thermal Engineering, 69, 106572.* DOI: 10.1016/j.csite.2025.106572

**被引：** 15次
**核心贡献：** 利用 Kern-Seaton 渐近模型 + GA-LM 混合反演方法，从工业数据推算渐近污垢热阻和时间常数。

---

## 七、工业应用案例

### 24. Gomez Suarez, Kennedy & Pugh et al. (2023)
"Fouling Management at TotalEnergies through Use of HTRI SmartPM: Case Study of a Project Proposal for Cleaning Schedule Optimization"
*Heat Transfer Engineering, 44(11), 941-951*

**被引：** 2次
**核心贡献：** TotalEnergies（道达尔）工业应用案例——使用商业软件 HTRI SmartPM 进行结垢监测和清洗调度优化的实际工程经验。有实际的经济效益数据。

---

## 总结：与概念图的对应关系

| 概念图节点 | 推荐重点论文（编号） |
|---|---|
| 结垢动力学 dRf/dt = φd − φr | #1, #2, #4, #6 |
| Rf(t) 演化曲线 | #4, #5, #8, #22 |
| 沉积项 φd（温度/浓度驱动） | #9, #10, #11, #17 |
| 剥离项 φr（剪切力/流速驱动） | #3, #9, #11 |
| 清洗阈值/优化决策 | #12, #13, #14, #15 |
| min{能耗损失+清洗费用} | #7, #12, #13 |
| 数据驱动的Rf预测 | #16–#21 |

**建议的阅读顺序：** #1 → #2 → #4 → #5 → #6 → #12 → #16 → #13

---

> 注意：以上论文的 DOI 和期刊信息是通过 OpenAlex 和 Semantic Scholar API 获取的。部分论文（尤其是 Kern & Seaton 1959 和 Epstein 1983 等经典论文）在这些数据库中收录不完整，建议通过 Google Scholar 或 ScienceDirect 直接搜索获取全文。
