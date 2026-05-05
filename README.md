# AI Homework Experiment 2

本项目是《人工智能导论》实验二的完整实现，任务是基于预训练词向量完成中文情感二分类，并比较多种神经网络模型在同一数据集上的表现。

当前仓库已经整理成“新同学拿到后可以直接运行”的结构。主程序默认会完成数据准备、主模型训练、附加对比实验、鲁棒性测试，并把结果写入 `outputs/`。

## 1. 项目目标

本项目围绕中文影评情感分类展开，核心目标包括：

- 实现并比较 `MLP`、`TextCNN`、`BiRNN`、`BiLSTM`、`BiGRU`
- 使用 50 维中文预训练 `Word2Vec` 词向量作为输入表示
- 统一训练、验证、测试流程，输出 `Accuracy`、`Precision`、`Recall`、`F1`
- 保存模型 checkpoint、实验结果汇总和后续分析文件

## 2. 数据集说明

### 2.1 主数据集

主数据位于：

- `data/train.txt`
- `data/validation.txt`
- `data/test.txt`

每一行格式为：

```text
label token1 token2 token3 ...
```

例如：

```text
1 这部 电影 很 感人
0 剧情 拖沓 表演 生硬
```

说明：

- `label` 是情感标签，当前任务为二分类
- 后面的内容已经是空格分词后的 token 序列
- 项目内部会把句子统一截断或补齐到长度 `80`

### 2.2 预训练词向量

词向量文件位于：

- `data/wiki_word2vec_50.bin`

这是主实验使用的输入表示来源，维度为 `50`。项目不会把整份词向量全部加载进内存，而是只抽取训练词表中实际出现的词向量。

### 2.3 外部鲁棒性测试集

外部测试数据位于：

- `data/external/weibo_sentiment_100.txt`
- `data/external/weibo_sentiment_100.jsonl`
- `data/external/weibo_sentiment_100_meta.json`

这部分数据用于测试模型跨域鲁棒性，不参与训练。

## 3. 当前项目架构

### 3.1 顶层目录

- `main.py`
  - 统一入口，直接运行主实验
- `scripts/`
  - 各类可执行脚本入口
- `src/sentiment_hw2/`
  - 核心实现代码
- `data/`
  - 训练、验证、测试数据和词向量
- `outputs/`
  - 训练结果、checkpoint、汇总 JSON
- `docs/`
  - 课程说明和提交材料
- `tools/`
  - 辅助检查脚本

### 3.2 核心代码分层

`src/sentiment_hw2/` 的职责大致如下：

- `data.py`
  - 读取数据集
  - 构建训练词表
  - 加载预训练词向量
  - 把文本编码成张量
- `models.py`
  - 定义 `MeanPoolMLP`、`TextCNN`、`BiRNNClassifier`、`BiLSTMClassifier`、`BiGRUClassifier`
- `train.py`
  - 单模型训练、验证、早停、指标计算、checkpoint 保存
- `experiment.py`
  - 主实验配置
  - 模型命名
  - 固定鲁棒性样本测试
- `paths.py`
  - 项目内所有常用路径常量
- `reporting.py` / `report_diagrams.py`
  - 报告与图表生成辅助代码

## 4. 输入和输出

### 4.1 主程序输入

主程序默认读取：

- `data/train.txt`
- `data/validation.txt`
- `data/test.txt`
- `data/wiki_word2vec_50.bin`

程序内部主要会得到以下张量：

- `inputs`
  - 形状约为 `[batch_size, 80]`
  - 表示 token id 序列
- `lengths`
  - 形状约为 `[batch_size]`
  - 表示真实长度
- `labels`
  - 形状约为 `[batch_size]`
  - 表示类别标签

### 4.2 模型输出

所有模型统一输出：

- `logits`
  - 形状为 `[batch_size, 2]`

训练时：

- 损失函数为 `CrossEntropyLoss`

预测时：

- 对 `logits` 做 `argmax` 得到类别

### 4.3 主程序输出

主实验运行后，主要输出位于 `outputs/`：

- `outputs/mlp_best.pt`
- `outputs/cnn_best.pt`
- `outputs/birnn_best.pt`
- `outputs/bilstm_best.pt`
- `outputs/bigru_best.pt`
- `outputs/transformer_best.pt`
- `outputs/*_metrics.json`
- `outputs/experiment_summary.json`

其中：

- `*_best.pt`
  - 保存该模型的最佳 checkpoint
- `*_metrics.json`
  - 保存该模型的训练结果、验证集和测试集指标、训练历史
- `experiment_summary.json`
  - 保存主实验总汇总，后续分析脚本也会依赖这个文件

## 5. 如何运行

### 5.1 安装依赖

最省事的方式是直接运行：

```bash
./setup.sh
```

它会自动：

- 创建 `.venv`
- 升级 `pip`
- 安装 `requirements.txt`

如果你想手动安装，也可以按下面方式执行：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

当前主要依赖：

- `torch==1.13.1`
- `numpy`
- `jieba`
- `reportlab`

### 5.2 直接运行主程序

完成环境安装后，最推荐的新手运行方式是：

```bash
./run.sh
```

或者：

```bash
bash run.sh
```

如果你已经明确希望手动指定 Python，也可以直接运行：

```bash
.venv/bin/python main.py
```

或者：

```bash
python3 main.py
```

前提是你已经安装好了依赖。

`run.sh` 会优先使用仓库内的 `.venv/bin/python`，找不到时再回退到系统 `python3` 或 `python`。它最终调用的是统一入口 `main.py`。`main.py` 内部直接调用主实验脚本 `scripts.run_experiments`，会自动完成：

1. 数据读取与缓存
2. 词表构建
3. 预训练词向量抽取
4. 主模型训练
5. 附加对比实验训练
6. 鲁棒性测试
7. 实验汇总保存

### 5.3 主程序等价命令

`main.py` 本质上等价于：

```bash
.venv/bin/python -m scripts.run_experiments
```

如果你更习惯模块方式，也可以直接运行它。

### 5.4 运行完成后的终端输出

主程序运行时会在终端打印：

- 当前设备（CPU 或 CUDA）
- 词表大小
- 训练语料被预训练词向量覆盖的比例
- 各模型的验证集和测试集指标
- 最终汇总文件保存位置

例如会看到类似输出：

```text
Device: cpu
Vocab size: 50000+
Train token coverage by pretrained vectors: 80%+
mlp: val_acc=..., val_f1=..., test_acc=..., test_f1=...
cnn: val_acc=..., val_f1=..., test_acc=..., test_f1=...
...
Summary saved to outputs/experiment_summary.json
```

## 6. 其他可选脚本

如果你只关心主实验，直接跑 `main.py` 就够了。

如果你还想继续扩展，可以使用下面这些入口：

### 6.1 单变量调参

```bash
.venv/bin/python -m scripts.run_hyperparameter_tuning
```

输出：

- `outputs/hyperparameter_tuning_summary.json`
- `outputs/tuning/`

### 6.2 逐步改进实验

```bash
.venv/bin/python -m scripts.run_improvement_experiments
```

输出：

- `outputs/improvement_experiment_summary.json`
- `outputs/improvement_results.csv`

### 6.3 Transformer 单独实验

```bash
.venv/bin/python -m scripts.run_transformer_experiment
```

输出：

- `outputs/transformer_metrics.json`
- `outputs/transformer_efficiency_comparison.json`

### 6.4 外部鲁棒性评估

```bash
.venv/bin/python -m scripts.evaluate_external_robustness
```

输出：

- `outputs/external_robustness_metrics.json`

## 7. 新人最小使用流程

如果你是第一次接触这个仓库，只需要按下面步骤操作：

1. 初始化环境

```bash
./setup.sh
```

2. 运行主实验

```bash
./run.sh
```

3. 查看输出结果

重点看：

- `outputs/experiment_summary.json`
- `outputs/*_metrics.json`
- `outputs/*_best.pt`

也就是说，新人不需要手动拼很多命令，直接运行 `./setup.sh` 和 `./run.sh` 就可以把主流程完整跑起来。

## 8. 当前默认实验模型

主程序默认训练以下模型：

- `MLP`
  - 先做 masked mean pooling，再做两层全连接分类
- `TextCNN`
  - 使用多卷积核提取局部 n-gram 特征
- `BiRNN`
  - 手写双向 vanilla RNN
- `BiLSTM`
  - 手写双向 LSTM
- `BiGRU`
  - 手写双向 GRU
- `Transformer`
  - 轻量 Transformer Encoder 基线

## 9. 目前默认实验设置

项目当前的一些关键默认设置如下：

- 最大句长：`80`
- 词向量维度：`50`
- 词表来源：仅训练集
- 主要评价指标：`Accuracy`、`Precision`、`Recall`、`F1`
- 主要早停指标：验证集 `F1`

## 10. 说明

- 本项目的主运行入口是 `main.py`
- 如果只想复现实验结果，不需要理解所有脚本，先跑主入口即可
- 结果分析、报告生成、改进实验、调参实验都是建立在主实验输出文件基础上的后续步骤
