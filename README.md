# AI Homework Experiment 2

本仓库完成《人工智能导论》实验二：基于预训练词向量的中文情感二分类。代码覆盖数据预处理、五类模型训练、指标汇总、图表导出和课程报告生成。

实现内容：

- `MLP baseline`
- `TextCNN`
- `BiRNN`
- `BiLSTM`
- `BiGRU`
- 统一训练、验证、测试与早停逻辑
- 参数对比实验与单变量调参实验
- 自动生成图表与实验报告 PDF

## 程序的输入和输出

### 1. 原始输入文件

- `train.txt` / `validation.txt` / `test.txt`
  - 每行格式：`label token1 token2 ...`
  - 例如：`1 这部 电影 很 感人`
- `wiki_word2vec_50.bin`
  - 50 维预训练词向量二进制文件
- `external_data/weibo_sentiment_100.txt`
  - 外部微博情感样本，供跨域鲁棒性测试使用

### 2. 送入模型前的输入张量

`src/sentiment_hw2/data.py` 会把文本样本编码成以下张量：

- `inputs`
  - 形状约为 `[batch_size, 80]`
  - 含义：每条文本被截断/补齐到长度 `80` 后的 token id 序列
- `lengths`
  - 形状约为 `[batch_size]`
  - 含义：每条文本的真实长度
- `labels`
  - 形状约为 `[batch_size]`
  - 含义：二分类标签 `0/1`

### 3. 模型输出

所有模型最终输出都是二分类 `logits`：

- 输出张量形状：`[batch_size, 2]`
- 第 1 维对应两类的未归一化分数
- 训练时使用 `CrossEntropyLoss`
- 推理时通常取 `argmax` 得到最终预测标签 `0/1`

### 4. 运行后的文件输出

训练和评测完成后，程序会在 `outputs/` 下写出这些产物：

- `outputs/*_best.pt`
  - 每个模型验证集最优轮次对应的 checkpoint
- `outputs/*_metrics.json`
  - 每个主模型的详细指标和完整训练历史
- `outputs/experiment_summary.json`
  - 主实验、参数对比实验和附加鲁棒性结果的总汇总
- `outputs/hyperparameter_tuning_summary.json`
  - 单变量调参实验汇总
- `outputs/external_robustness_metrics.json`
  - 外部测试集的评测结果
- `outputs/cache/prepared_data.pt`
  - 预处理缓存，重复运行时会复用
- `学号_姓名.pdf`
  - 最终课程报告

## 有没有损失函数和图表输出

有，而且已经在仓库里生成了一部分结果图。

`generate_report.py` 会根据 `outputs/experiment_summary.json` 和外部评测结果自动导出图表到 `outputs/report_assets/`。当前仓库里可见的典型图表包括：

- `mlp_loss_curve.png`
- `cnn_loss_curve.png`
- `birnn_loss_curve.png`
- `bilstm_loss_curve.png`
- `bigru_loss_curve.png`
- `validation_f1_curve.png`
- `test_metric_bar_chart.png`
- `external_robustness_chart.png`

这些图分别表示：

- 各主模型的 `train_loss` / `validation_loss` 曲线
- 五个主模型在验证集上的 `F1` 收敛曲线
- 五个主模型在测试集上的 `Accuracy` 和 `F1` 柱状图
- 原测试集与外部微博测试集之间的鲁棒性对比图

如果你关心“损失函数有没有输出”，答案是有：

- 训练阶段使用的是 `CrossEntropyLoss`
- 每一轮的 `train_loss` 和 `validation_loss` 都会写入各模型的 `*_metrics.json`
- 报告脚本会把这些 loss 历史进一步画成曲线图

## 如何运行

### 1. 安装依赖

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. 运行主实验

```bash
.venv/bin/python run_experiments.py
```

这一步会完成：

- 数据读取与编码
- 训练 `MLP / TextCNN / BiRNN / BiLSTM / BiGRU`
- 在验证集上早停
- 在测试集上输出最终指标
- 保存 checkpoint 和 `outputs/experiment_summary.json`

### 3. 生成报告和图表

```bash
.venv/bin/python generate_report.py
```

这一步会：

- 读取 `outputs/experiment_summary.json`
- 导出训练曲线、F1 曲线、柱状图等图表到 `outputs/report_assets/`
- 生成最终报告 `学号_姓名.pdf`

### 4. 可选：运行单变量调参实验

```bash
.venv/bin/python run_hyperparameter_tuning.py
```

这一步会额外生成：

- `outputs/tuning/*.pt`
- `outputs/tuning/*_metrics.json`
- `outputs/hyperparameter_tuning_summary.json`

### 5. 可选：外部鲁棒性测试

如果需要重新抓取或整理外部样本：

```bash
.venv/bin/python fetch_external_samples.py
```

然后运行：

```bash
.venv/bin/python evaluate_external_robustness.py
```

这一步会生成：

- `outputs/external_robustness_metrics.json`

## 目录说明

- `run_experiments.py`：主实验入口，串行执行数据准备、模型训练、对比实验和结果汇总
- `generate_report.py`：报告入口，读取汇总结果并生成 `学号_姓名.pdf`
- `run_hyperparameter_tuning.py`：逐个超参数做单变量调参实验
- `fetch_external_samples.py`：抓取并构造外部鲁棒性测试集
- `evaluate_external_robustness.py`：加载已训练模型并评测外部测试集
- `src/sentiment_hw2/data.py`：读取数据、构建词表、抽取词向量、编码样本
- `src/sentiment_hw2/models.py`：定义 `MLP / TextCNN / BiRNN / BiLSTM / BiGRU`
- `src/sentiment_hw2/train.py`：训练循环、评估指标、早停和 checkpoint 保存
- `src/sentiment_hw2/experiment.py`：主实验配置、对比实验配置、鲁棒性样本和推理辅助
- `src/sentiment_hw2/reporting.py`：报告图表、表格和版式辅助函数
- `outputs/`：训练产物、指标汇总、缓存和图表输出目录

## 主要实现细节

- 词表只使用训练集构建，避免验证集和测试集信息泄漏
- 句长统一截断或补齐到 `80`
- 评价指标包含 `Accuracy`、`Precision`、`Recall`、`F1`
- 以验证集 `F1` 为主进行早停，并保存最优参数
- 训练阶段统一记录每轮 `train_loss`、`validation_loss`、`validation_accuracy`、`validation_f1`
- 报告图表全部由程序根据实验结果自动生成，不需要手工画图
