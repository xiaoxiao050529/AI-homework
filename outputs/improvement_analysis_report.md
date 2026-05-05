# CNN / RNN 改进实验分析报告

生成时间：2026-05-04

## 报告说明
这份报告单独分析 `run_improvement_experiments.py` 产出的逐步改进实验，不替代原课程总报告。
本轮新增训练采用 `quick` 预算以适应当前 CPU 环境：新增 RNN 变体使用较短训练轮数与更大 batch；`BiRNN / BiLSTM / TextCNN` 三个基线直接复用了主实验已有结果。
因此，这里更适合用于判断“改进方向是否值得继续”，如果你要把某个新变体写成最终主结论，建议再用完整预算复验一次。

## 数据与评估
训练集 / 验证集 / 测试集规模分别为 `19998` / `5629` / `369`，最大句长 `80`，词表大小 `53339`。
统一指标为 `Accuracy` 与 `F1`，主要关注测试集 F1 的变化。

## RNN 路线
| 实验 | 训练数据处理 | Val Acc | Val F1 | Test Acc | Test F1 | 相对上一阶段 Test F1 | 结果来源 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| BiRNN 基线 | none | 0.8422 | 0.8462 | 0.8401 | 0.8384 | - | 复用已有主实验 |
| BiLSTM | none | 0.8428 | 0.8477 | 0.8618 | 0.8625 | +0.0242 | 复用已有主实验 |
| BiLSTM + Attention | none | 0.7689 | 0.7845 | 0.7967 | 0.8011 | -0.0615 | 本次新增训练 |

结论：`BiLSTM` 相比 `BiRNN 基线` 的测试集 F1 提升了 `+0.0242`，说明把普通循环单元换成 LSTM 在这份影评数据上是有效的。
结论：`BiLSTM + Attention` 在本轮 quick 预算下测试集 F1 相比 `BiLSTM` 下降了 `-0.0615`。这更像是“新增注意力结构需要继续调参”而不是“注意力一定无效”，因为它本次使用的是更短训练预算，且没有单独调注意力维度、学习率与 dropout。
本路线当前测试集 F1 最好的是 `BiLSTM`，分数为 `0.8625`。

## CNN 路线
| 实验 | 训练数据处理 | Val Acc | Val F1 | Test Acc | Test F1 | 相对上一阶段 Test F1 | 结果来源 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| CNN 单卷积核基线 | none | 0.8206 | 0.8139 | 0.8266 | 0.8140 | - | 本次新增训练 |
| TextCNN 多卷积核 | none | 0.8424 | 0.8422 | 0.8537 | 0.8516 | +0.0377 | 复用已有主实验 |
| TextCNN + Random Deletion | random_deletion | 0.8394 | 0.8406 | 0.8320 | 0.8297 | -0.0220 | 本次新增训练 |
| TextCNN + Synonym Replacement | synonym_replacement | 0.8399 | 0.8408 | 0.8347 | 0.8310 | -0.0206 | 本次新增训练 |
| TextCNN + Word Dropout | word_dropout | 0.8389 | 0.8423 | 0.8401 | 0.8392 | -0.0124 | 本次新增训练 |

结论：`TextCNN 多卷积核` 相比 `CNN 单卷积核基线` 的测试集 F1 提升了 `+0.0377`，说明多卷积核 TextCNN 对不同长度情感短语的建模收益很明显。
结论：三种轻量增强都没有超过原始 `TextCNN 多卷积核`。其中损失最小的是 `TextCNN + Word Dropout`，测试集 F1 相比基线变动 `-0.0124`；`Random Deletion` 与 `Synonym Replacement` 的退化更明显，说明当前增强强度对这份数据来说偏大。
本路线当前测试集 F1 最好的是 `TextCNN 多卷积核`，分数为 `0.8516`。

## 总结
1. 在当前代码和数据上，`BiRNN -> BiLSTM` 是明确有效的升级，测试集 F1 提升约 `+0.0242`。
2. `BiLSTM + Attention` 这次没有跑赢 `BiLSTM`，但它属于新增结构且只做了 quick 预算，后续还有继续调参空间。
3. `单卷积核 CNN -> TextCNN 多卷积核` 的收益最稳定，是本轮 CNN 路线里最值得保留到正式报告的结构改进。
4. 三种轻量增强里，`Word Dropout` 最稳，但在这次设置下仍未超过未增强 TextCNN；说明增强存在“强度过大或改写噪声过强”的问题。

## 建议下一步
1. 把 `BiLSTM + Attention` 用完整预算再跑一次，并单独搜索 `attention_dim`、`dropout` 和 `learning_rate`。
2. 保留 `TextCNN 多卷积核` 作为 CNN 主结果，再把 `Word Dropout` 的概率从 `0.1` 下调到 `0.05` 重新验证。
3. 如果你还想继续扩展 RNN，可以追加 `BiLSTM + Max/Mean Pooling`，它通常比 Attention 更容易稳定提升。

## 附件
- 明细结果表：`outputs/improvement_results.csv`
- 原始汇总 JSON：`outputs/improvement_experiment_summary.json`
