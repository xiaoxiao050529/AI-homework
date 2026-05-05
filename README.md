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

## 3. 当前代码架构

这一版仓库已经不是“单文件脚本”，而是按“入口层 -> 核心库层 -> 实验脚本层 -> 输出产物层”组织的。理解这一层之后，再看各个 `.py` 文件会快很多。

### 3.1 总体分层

- `main.py`
  - 最薄的一层入口
  - 只负责把执行权交给 `scripts.run_experiments.main()`
- `scripts/`
  - 面向命令行的实验入口层
  - 每个脚本负责一个完整任务，例如主实验、调参、改进实验、报告生成、tmux 后台运行
- `src/sentiment_hw2/`
  - 核心库层
  - 真正的可复用逻辑都放在这里，包括数据处理、模型定义、训练循环、实验配置、报告绘图、增强策略
- `data/`
  - 原始输入数据层
  - 包含训练/验证/测试集、预训练词向量，以及外部鲁棒性测试样本
- `outputs/`
  - 实验产物层
  - 包含 checkpoint、metrics、汇总 JSON、报告素材、调参结果、缓存文件
- `docs/`
  - 课程说明与提交说明
- `tools/`
  - 辅助检查脚本

### 3.2 主流程执行链路

默认运行 `./run.sh` 时，实际执行链路是：

```text
run.sh
  -> main.py
    -> scripts/run_experiments.py
      -> prepare_data(...)              # 读取数据、建词表、抽取词向量、缓存 prepared_data.pt
      -> build_main_model_specs(...)    # 生成主实验模型配置
      -> train_single_model(...)        # 逐个模型训练、早停、保存 checkpoint 和 metrics
      -> build_ablation_specs(...)      # 运行附加对比实验
      -> run_robustness(...)            # 对已训练主模型做固定样本鲁棒性检查
      -> outputs/experiment_summary.json
```

也就是说，主入口本身几乎不写业务逻辑，核心工作都被拆到了 `src/sentiment_hw2/` 里。

### 3.3 核心库层 `src/sentiment_hw2/`

#### `paths.py`

- 统一维护项目中的常用路径常量
- 例如 `TRAIN_PATH`、`VALIDATION_PATH`、`TEST_PATH`、`EMBEDDING_PATH`、`OUTPUT_DIR`
- 通过 `_prefer_existing(...)` 兼容“文件放在标准目录”与“文件放在项目根目录”两种情况

#### `data.py`

负责整个数据准备链路：

- `read_split()`
  - 读取 `label token1 token2 ...` 格式的数据
- `build_vocab()`
  - 只基于训练集构建词表，避免验证集和测试集泄漏
- `load_word2vec_subset()`
  - 从大词向量二进制文件中只抽取当前词表真正需要的词
- `_encode_samples()`
  - 把 token 序列截断/补齐到固定长度，并编码成张量
- `_build_embedding_matrix()`
  - 用预训练向量初始化词表；OOV 词使用同分布随机向量
- `prepare_data()`
  - 串起整条数据准备流程
  - 会把结果缓存到 `outputs/cache/prepared_data.pt`

这里产出的 `PreparedData` 是后续所有实验共享的标准输入对象。

#### `models.py`

负责模型定义与参数初始化，当前仓库里的核心模型都在这里：

- `MeanPoolMLP`
  - `embedding -> masked mean pooling -> MLP classifier`
- `TextCNN`
  - 多卷积核提取局部 n-gram 特征
- `TransformerEncoderClassifier`
  - 轻量 Transformer Encoder 基线
- `BiRNNClassifier` / `BiLSTMClassifier` / `BiGRUClassifier`
  - 双向循环模型分类器

这里有一个很关键的实现特点：

- 循环模型不是完全依赖框架现成封装
- 文件中实现了 `ManualRNNCell`、`ManualLSTMCell`、`ManualGRUCell`
- 再由 `ManualBidirectionalRecurrentEncoder` 组织成双向多层编码器

因此，这个仓库不只是“调用 PyTorch 现成层跑实验”，而是保留了一层手写循环单元实现，方便课程展示模型内部结构。

#### `train.py`

负责统一训练框架，是整个实验体系的核心调度模块之一：

- `TrainConfig`
  - 单个实验的训练超参数配置
- `create_dataloaders()`
  - 为 train / validation / test 构建 `DataLoader`
- `compute_metrics()`
  - 统一计算 `Accuracy`、`Precision`、`Recall`、`F1`
- `evaluate()`
  - 在验证集/测试集上做完整评估
- `train_single_model()`
  - 训练单个模型
  - 支持 embedding 单独学习率
  - 支持 label smoothing
  - 支持 warmup + cosine 或 plateau scheduler
  - 支持冻结 embedding 若干轮
  - 支持早停、checkpoint 保存、history 记录

项目里大多数实验脚本最后都会落到 `train_single_model()` 上，因此这部分相当于是“统一训练引擎”。

#### `experiment.py`

负责“实验编排”，而不是底层训练：

- `ModelSpec`
  - 把“模型构造器 + 训练配置 + 名称”打包
- `build_main_model_specs()`
  - 定义主实验默认训练的模型列表
- `build_ablation_specs()`
  - 定义附加对比实验
- `build_transformer_spec()`
  - 单独生成 Transformer 配置
- `run_robustness()`
  - 对固定样本集做统一推理检查

换句话说：

- `models.py` 决定“模型长什么样”
- `train.py` 决定“怎么训练”
- `experiment.py` 决定“这一轮到底训练哪些模型、用什么超参数”

#### `augment.py`

负责训练集增强，仅服务于改进实验：

- `word_dropout`
- `random_deletion`
- `synonym_replacement`
- `augment_prepared_training_data`

这些增强只改训练切分，不改验证集和测试集。

#### `reporting.py` / `report_diagrams.py`

负责课程报告输出：

- 生成表格、统计图、训练曲线、结构图
- 输出到 `outputs/report_assets/`
- 被 `scripts/generate_report.py` 调用，用于最终 PDF 报告拼装

### 3.4 脚本层 `scripts/`

这一层的特点是：每个脚本基本都对应一个“完整任务”。

#### 主实验与评估

- `run_experiments.py`
  - 主实验总入口
  - 训练主模型、附加对比实验，并生成 `experiment_summary.json`
- `run_transformer_experiment.py`
  - 单独训练 Transformer，并和已有主模型结果做效率对比
- `evaluate_external_robustness.py`
  - 使用外部数据集评测已训练模型的跨域鲁棒性

#### 调参与改进实验

- `run_hyperparameter_tuning.py`
  - 五类模型的单变量调参
  - 输出 `outputs/tuning/` 与 `hyperparameter_tuning_summary.json`
- `run_recurrent_tuning_pipeline.py`
  - 按参数顺序依次跑循环模型调参
- `run_recurrent_tuning_parallel.py`
  - 并行启动 `BiRNN` / `BiLSTM` / `BiGRU` 调参
- `run_improvement_experiments.py`
  - 跑“逐步改进”路线，例如 `BiLSTM + Attention`、`TextCNN + 数据增强`
- `run_stopping_strategy_experiments.py`
  - 对比固定轮数训练与早停策略

#### 报告与数据辅助

- `generate_report.py`
  - 基于已有结果生成课程 PDF 报告
- `generate_improvement_report.py`
  - 根据 staged improvement 结果生成分析文档
- `fetch_external_samples.py`
  - 抓取并构造外部鲁棒性测试样本

#### 长任务运行辅助

- `run_in_tmux.py`
  - 在 tmux 会话里后台运行长实验
  - 日志写入 `outputs/tmux/logs/`
  - 会话元信息写入 `outputs/tmux/sessions/`

### 3.5 当前产物目录 `outputs/`

当前 `outputs/` 已经不是单一结果目录，而是按用途分层的：

- `outputs/*_best.pt`
  - 单个实验的最佳 checkpoint
- `outputs/*_metrics.json`
  - 单个实验的训练/验证/测试指标与历史
- `outputs/experiment_summary.json`
  - 主实验总汇总
- `outputs/external_robustness_metrics.json`
  - 外部鲁棒性评估结果
- `outputs/hyperparameter_tuning_summary.json`
  - 调参汇总
- `outputs/improvement_experiment_summary.json`
  - 改进实验汇总
- `outputs/transformer_efficiency_comparison.json`
  - Transformer 与主模型的效率对比
- `outputs/cache/`
  - 数据缓存与 HuggingFace 特征缓存
- `outputs/tuning/`
  - 单变量调参产物
- `outputs/report_assets/`
  - 报告用图表和结构图
- `outputs/recheck_current_main/`
  - 一轮重检主实验的独立结果目录

### 3.6 看代码时的推荐顺序

如果你想快速理解整个仓库，推荐按下面顺序阅读：

1. `main.py`
2. `scripts/run_experiments.py`
3. `src/sentiment_hw2/experiment.py`
4. `src/sentiment_hw2/train.py`
5. `src/sentiment_hw2/data.py`
6. `src/sentiment_hw2/models.py`
7. 再按需看 `augment.py`、`reporting.py`、其他 `scripts/`

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
