# 家庭电力消耗多变量时间序列预测实验报告

## 1. 问题介绍

本项目研究家庭电力消耗预测问题。数据来自 UCI Individual Household Electric Power Consumption，原始记录以分钟为粒度，包含全屋有功功率、无功功率、电压、电流以及三个子表能耗。按照课程要求，实验将分钟数据汇总为日级数据，其中 `global_active_power`、`global_reactive_power`、`sub_metering_1`、`sub_metering_2`、`sub_metering_3` 按天求和，`voltage` 和 `global_intensity` 按天求平均，并补充 `sub_metering_remainder` 与日期周期特征。

预测任务为使用过去 90 天的多变量序列预测未来总有功功率曲线，分别设置 90 天短期预测和 365 天长期预测。两种预测长度分别训练模型，评价指标为 MSE 和 MAE。每个模型在 5 个随机种子下重复实验，并报告均值和标准差。

## 2. 模型

### 2.1 LSTM

LSTM 将 90 天多变量序列按时间步输入循环网络，使用最后一层隐藏状态作为历史用电模式的表示，再通过全连接层直接输出未来 `horizon` 天的预测序列。该模型适合捕获局部趋势和中短期时序依赖。

### 2.2 Transformer

Transformer 首先将每日特征投影到隐藏维度，并加入正弦位置编码；随后使用多头自注意力编码 90 天历史序列，最后取最后一个时间步的表示输出未来曲线。相比 LSTM，Transformer 对长距离依赖建模更直接，但在小样本时间序列上更依赖正则化。

### 2.3 改进模型：CNN-Transformer

本文提出 CNN-Transformer 组合模型。模型先用一维卷积在时间轴上提取短期局部模式，例如连续几天的用电波动和周期片段；随后将卷积后的序列输入 Transformer 编码器建模长期依赖。最后使用均值池化表示与最后时间步表示的门控融合，输出未来曲线。该结构的动机是让卷积层承担局部平滑和局部模式抽取，减少 Transformer 在小样本长期预测中的学习难度。

简化伪代码如下：

```text
X = daily_features[t-90:t]
Z = Conv1D(X)
H = TransformerEncoder(PositionalEncoding(Z))
context = mean_pool(H) * sigmoid(W * H_last)
y_hat = Linear(LayerNorm(context))
```

## 3. 结果与分析

五轮实验的均值和标准差如下：

| Model | Horizon | MSE mean | MSE std | MAE mean | MAE std |
|---|---:|---:|---:|---:|---:|
| cnn_transformer | 90 | 131725.2016 | 4237.5402 | 284.9127 | 5.7690 |
| cnn_transformer | 365 | 179540.6125 | 5436.1238 | 310.7817 | 2.7848 |
| lstm | 90 | 138870.3781 | 12303.6988 | 296.1077 | 12.6983 |
| lstm | 365 | 194343.8719 | 8947.0416 | 327.1796 | 9.0630 |
| transformer | 90 | 124860.4469 | 5981.2597 | 283.1321 | 4.6442 |
| transformer | 365 | 180228.6000 | 5004.4971 | 310.5392 | 5.4646 |

预测曲线如下：

![LSTM 90-day](../outputs/figures/lstm_90.png)
![Transformer 90-day](../outputs/figures/transformer_90.png)
![CNN-Transformer 90-day](../outputs/figures/cnn_transformer_90.png)
![LSTM 365-day](../outputs/figures/lstm_365.png)
![Transformer 365-day](../outputs/figures/transformer_365.png)
![CNN-Transformer 365-day](../outputs/figures/cnn_transformer_365.png)

从任务性质看，90 天预测通常更容易保持趋势和幅值稳定；365 天预测需要跨季节建模，误差会明显增大。若 CNN-Transformer 在长期预测上优于纯 Transformer，说明局部卷积特征对稳定长期依赖学习有帮助；若性能不佳，可能原因是数据量较小、家庭行为模式突变、以及 365 天直接多步输出对模型容量和正则化更敏感。

## 4. 讨论

本实验使用直接多步预测，而不是递归单步预测，因此避免了递归误差逐步累积，但要求模型一次性学习完整未来曲线。对家庭电力数据而言，日期周期特征可以提供星期和月份信息，帮助模型拟合生活规律和季节性变化。后续改进可以加入真实气象数据、节假日特征，或采用分解式预测方法先建模趋势与季节项，再预测残差。

本报告撰写过程中使用了 ChatGPT/Codex 辅助整理文字和代码结构；模型设计、实验运行与结果分析仍需以仓库中的可复现实验输出为准。

## 参考文献

[1] UCI Machine Learning Repository. Individual household electric power consumption. https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

[2] Vaswani, A. et al. Attention Is All You Need. NeurIPS, 2017.

[3] Hochreiter, S., Schmidhuber, J. Long Short-Term Memory. Neural Computation, 1997.

[4] 课程作业说明《2026年专硕机器学习课程项目》。
