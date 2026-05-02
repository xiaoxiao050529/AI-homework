# AI Homework Experiment 2

本仓库完成《人工智能导论》实验二：基于预训练词向量的中文情感分类。

实现内容：

- `MLP baseline`
- `TextCNN`
- `BiRNN`
- `BiLSTM`
- `BiGRU`
- 统一训练、验证、测试与早停逻辑
- 参数对比实验
- 自动生成实验报告 PDF

## 目录

- `run_experiments.py`：实验入口，负责串行执行数据准备、主实验、消融实验和汇总
- `generate_report.py`：报告入口，读取汇总结果并生成 `实验二报告.pdf`
- `fetch_external_samples.py`：抓取并构造外部鲁棒性测试集
- `evaluate_external_robustness.py`：加载已训练模型并评测外部测试集
- `src/sentiment_hw2/data.py`：读取数据、构建词表、抽取词向量、编码样本
- `src/sentiment_hw2/models.py`：定义 `MLP / TextCNN / BiRNN / BiLSTM / BiGRU` 等模型与参数初始化
- `src/sentiment_hw2/train.py`：训练循环、评估指标、早停和 checkpoint 保存
- `src/sentiment_hw2/experiment.py`：主实验配置、消融实验配置、鲁棒性样本和推理辅助
- `src/sentiment_hw2/report_diagrams.py`：生成报告中使用的模型结构图位图
- `src/sentiment_hw2/reporting.py`：报告样式、表格、图片插入和章节辅助函数
- `outputs/`：训练日志、指标汇总、缓存和报告图片资源

## 运行方式

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_experiments.py
.venv/bin/python generate_report.py
```

外部鲁棒性测试：

```bash
.venv/bin/python fetch_external_samples.py
.venv/bin/python evaluate_external_robustness.py
```

## 数据说明

- `train.txt` / `validation.txt` / `test.txt`：每行格式为 `label token1 token2 ...`
- `wiki_word2vec_50.bin`：50 维预训练词向量
- `external_data/weibo_sentiment_100.txt`：抓取并分词后的外部微博情感样本，用于跨域鲁棒性测试

## 实现细节

- 词表仅使用训练集构建，避免测试集信息泄漏
- 句长统一截断或补齐到 80
- 评价指标包含 `Accuracy`、`Precision`、`Recall` 和 `F1`
- 以验证集 `F1` 为主进行早停，并保存最优参数
- 额外提供外部小规模数据评测脚本，用于比较模型在跨域评论上的鲁棒性

## 当前代码结构

- 入口脚本尽量保持薄：训练入口只做流程编排，报告入口只做章节编排
- 可复用逻辑统一下沉到 `src/sentiment_hw2/`，避免脚本里堆放配置、绘图和工具函数
- 主实验口径统一为 `MLP / TextCNN / BiRNN / BiLSTM / BiGRU`，与报告内容保持一致
