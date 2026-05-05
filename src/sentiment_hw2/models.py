"""课程实验中使用的几类文本分类模型。"""

from typing import Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


RecurrentState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def _trainable_embedding(embedding_matrix: torch.Tensor) -> nn.Embedding:
    """为每个模型创建独立词向量副本，避免多次实验共享可训练权重。"""
    return nn.Embedding.from_pretrained(
        embedding_matrix.detach().clone(), freeze=False, padding_idx=0
    )


class MeanPoolMLP(nn.Module):
    """先做 masked mean pooling，再用两层感知机分类。"""

    def __init__(self, embedding_matrix: torch.Tensor, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.embedding = _trainable_embedding(embedding_matrix)
        embedding_dim = embedding_matrix.size(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """忽略 PAD 位置后对词向量取平均，得到句向量。"""
        embedded = self.embedding(inputs)
        # 通过 mask 去掉补齐位置，避免平均池化被大量 PAD 稀释。
        mask = (inputs != 0).unsqueeze(-1).float()
        summed = (embedded * mask).sum(dim=1)
        denom = lengths.clamp(min=1).unsqueeze(1).float()
        pooled = summed / denom
        return self.classifier(self.dropout(pooled))


class TextCNN(nn.Module):
    """典型 TextCNN：多尺寸卷积核提取局部 n-gram 模式。"""

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        num_filters: int = 128,
        filter_sizes: Sequence[int] = (3, 4, 5),
        dropout: float = 0.5,
    ):
        super().__init__()
        self.embedding = _trainable_embedding(embedding_matrix)
        embedding_dim = embedding_matrix.size(1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (kernel_size, embedding_dim)) for kernel_size in filter_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), 2)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """在时间维做卷积和最大池化后进行二分类。"""
        del lengths
        embedded = self.embedding(inputs).unsqueeze(1)
        # 每个卷积核尺寸对应一组局部短语特征图。
        conved = [torch.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [torch.max(item, dim=2).values for item in conved]
        features = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(features))


class TransformerEncoderClassifier(nn.Module):
    """基于 Transformer Encoder 的文本分类器。"""

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.2,
        max_len: int = 80,
        pooling: str = "cls",
    ):
        super().__init__()
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be 'cls' or 'mean'")
        self.embedding = _trainable_embedding(embedding_matrix)
        embedding_dim = embedding_matrix.size(1)
        self.input_proj = nn.Linear(embedding_dim, model_dim)
        self.position_embedding = nn.Embedding(max_len + 1, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.input_norm = nn.LayerNorm(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(model_dim, 2)
        self.max_len = max_len
        self.pooling = pooling

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """对序列做自注意力编码，再对有效 token 表示做均值池化。"""
        batch_size, seq_len = inputs.size()
        seq_len = min(seq_len, self.max_len)
        effective_lengths = lengths.clamp(min=1, max=seq_len)
        embedded = self.embedding(inputs[:, :seq_len])

        positions = torch.arange(seq_len + 1, device=inputs.device).unsqueeze(0).expand(batch_size, seq_len + 1)
        token_hidden = self.input_proj(embedded)
        cls_hidden = self.cls_token.expand(batch_size, -1, -1)
        hidden = torch.cat([cls_hidden, token_hidden], dim=1) + self.position_embedding(positions)
        hidden = self.input_norm(hidden)

        token_mask = torch.arange(seq_len, device=inputs.device).unsqueeze(0).expand(batch_size, seq_len) >= effective_lengths.unsqueeze(1)
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=inputs.device)
        padding_mask = torch.cat([cls_mask, token_mask], dim=1)
        encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)

        if self.pooling == "cls":
            pooled = encoded[:, 0, :]
        else:
            token_encoded = encoded[:, 1:, :]
            valid_mask = (~token_mask).unsqueeze(-1).float()
            pooled = (token_encoded * valid_mask).sum(dim=1) / effective_lengths.unsqueeze(1).float()
        return self.fc(self.dropout(pooled))


class ManualRNNCell(nn.Module):
    """只用张量运算实现的 vanilla RNN 单元。"""

    def __init__(self, input_size: int, hidden_size: int, nonlinearity: str = "tanh"):
        super().__init__()
        if nonlinearity not in {"tanh", "relu"}:
            raise ValueError("nonlinearity must be 'tanh' or 'relu'")
        self.weight_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(hidden_size))
        self.activation = torch.tanh if nonlinearity == "tanh" else torch.relu

    def forward(self, input_t: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        pre_activation = F.linear(input_t, self.weight_ih, self.bias_ih)
        pre_activation = pre_activation + F.linear(hidden, self.weight_hh, self.bias_hh)
        return self.activation(pre_activation)


class ManualGRUCell(nn.Module):
    """只用张量运算实现的 GRU 单元。"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        gate_size = hidden_size * 3
        self.weight_ih = nn.Parameter(torch.empty(gate_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(gate_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(gate_size))
        self.bias_hh = nn.Parameter(torch.empty(gate_size))

    def forward(self, input_t: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        input_linear = F.linear(input_t, self.weight_ih, self.bias_ih)
        hidden_linear = F.linear(hidden, self.weight_hh, self.bias_hh)
        input_reset, input_update, input_new = input_linear.chunk(3, dim=1)
        hidden_reset, hidden_update, hidden_new = hidden_linear.chunk(3, dim=1)

        reset_gate = torch.sigmoid(input_reset + hidden_reset)
        update_gate = torch.sigmoid(input_update + hidden_update)
        new_hidden = torch.tanh(input_new + reset_gate * hidden_new)
        return (1.0 - update_gate) * new_hidden + update_gate * hidden


class ManualLSTMCell(nn.Module):
    """只用张量运算实现的 LSTM 单元。"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        gate_size = hidden_size * 4
        self.weight_ih = nn.Parameter(torch.empty(gate_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(gate_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(gate_size))
        self.bias_hh = nn.Parameter(torch.empty(gate_size))

    def forward(
        self,
        input_t: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden, cell = state
        gates = F.linear(input_t, self.weight_ih, self.bias_ih)
        gates = gates + F.linear(hidden, self.weight_hh, self.bias_hh)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=1)

        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        output_gate = torch.sigmoid(output_gate)

        next_cell = forget_gate * cell + input_gate * cell_gate
        next_hidden = output_gate * torch.tanh(next_cell)
        return next_hidden, next_cell


class ManualBidirectionalRecurrentEncoder(nn.Module):
    """手写双向多层循环编码器，替代框架封装的循环层。"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        cell_type: str,
        nonlinearity: str = "tanh",
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout
        self.dropout = nn.Dropout(dropout)
        self.cell_type = cell_type
        self.nonlinearity = nonlinearity
        self.forward_cells = nn.ModuleList()
        self.backward_cells = nn.ModuleList()

        layer_input_size = input_size
        for _ in range(num_layers):
            self.forward_cells.append(self._build_cell(layer_input_size))
            self.backward_cells.append(self._build_cell(layer_input_size))
            layer_input_size = hidden_size * 2

    def _build_cell(self, input_size: int) -> nn.Module:
        if self.cell_type == "rnn":
            return ManualRNNCell(input_size, self.hidden_size, self.nonlinearity)
        if self.cell_type == "gru":
            return ManualGRUCell(input_size, self.hidden_size)
        if self.cell_type == "lstm":
            return ManualLSTMCell(input_size, self.hidden_size)
        raise ValueError("Unsupported cell_type: {}".format(self.cell_type))

    def _zero_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> RecurrentState:
        zeros = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        if self.cell_type == "lstm":
            return zeros, torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        return zeros

    @staticmethod
    def _hidden_from_state(state: RecurrentState) -> torch.Tensor:
        if isinstance(state, tuple):
            return state[0]
        return state

    def _run_direction(
        self,
        cell: nn.Module,
        inputs: torch.Tensor,
        lengths: torch.Tensor,
        reverse: bool,
    ) -> Tuple[torch.Tensor, RecurrentState]:
        batch_size, seq_len, _ = inputs.size()
        state = self._zero_state(batch_size, inputs.device, inputs.dtype)
        outputs = inputs.new_zeros(batch_size, seq_len, self.hidden_size)
        steps = range(seq_len - 1, -1, -1) if reverse else range(seq_len)

        for step in steps:
            previous_state = state
            current_input = inputs[:, step, :]
            current_state = cell(current_input, previous_state)
            active = (lengths > step).unsqueeze(1).type_as(inputs)
            inactive = 1.0 - active

            if isinstance(current_state, tuple):
                current_hidden, current_cell = current_state
                previous_hidden, previous_cell = previous_state
                next_hidden = current_hidden * active + previous_hidden * inactive
                next_cell = current_cell * active + previous_cell * inactive
                state = (next_hidden, next_cell)
            else:
                state = current_state * active + previous_state * inactive

            outputs[:, step, :] = self._hidden_from_state(state)

        return outputs, state

    def forward(
        self,
        inputs: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        lengths = lengths.to(inputs.device)
        layer_input = inputs
        final_hidden_states = []
        final_cell_states = []

        for layer_index, (forward_cell, backward_cell) in enumerate(
            zip(self.forward_cells, self.backward_cells)
        ):
            forward_outputs, forward_state = self._run_direction(
                forward_cell, layer_input, lengths, reverse=False
            )
            backward_outputs, backward_state = self._run_direction(
                backward_cell, layer_input, lengths, reverse=True
            )
            layer_output = torch.cat([forward_outputs, backward_outputs], dim=2)
            if layer_index < self.num_layers - 1 and self.dropout_prob > 0.0:
                layer_output = self.dropout(layer_output)

            final_hidden_states.extend(
                [self._hidden_from_state(forward_state), self._hidden_from_state(backward_state)]
            )
            if self.cell_type == "lstm":
                final_cell_states.extend([forward_state[1], backward_state[1]])
            layer_input = layer_output

        hidden = torch.stack(final_hidden_states, dim=0)
        if self.cell_type == "lstm":
            cell = torch.stack(final_cell_states, dim=0)
            return layer_input, (hidden, cell)
        return layer_input, hidden


class BiGRUClassifier(nn.Module):
    """双向 GRU 文本分类器，循环单元完全手写。"""

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = _trainable_embedding(embedding_matrix)
        embedding_dim = embedding_matrix.size(1)
        self.gru = ManualBidirectionalRecurrentEncoder(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            cell_type="gru",
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """取双向 GRU 最后一层的前向和后向隐藏状态作为句向量。"""
        embedded = self.embedding(inputs)
        _, hidden = self.gru(embedded, lengths)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        features = torch.cat([forward_hidden, backward_hidden], dim=1)
        return self.fc(self.dropout(features))


class BiRNNClassifier(nn.Module):
    """双向普通 RNN 文本分类器，循环单元完全手写。"""

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = _trainable_embedding(embedding_matrix)
        embedding_dim = embedding_matrix.size(1)
        self.rnn = ManualBidirectionalRecurrentEncoder(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            cell_type="rnn",
            nonlinearity="tanh",
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """拼接双向 RNN 的最终隐藏状态后完成分类。"""
        embedded = self.embedding(inputs)
        _, hidden = self.rnn(embedded, lengths)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        features = torch.cat([forward_hidden, backward_hidden], dim=1)
        return self.fc(self.dropout(features))


class BiLSTMClassifier(nn.Module):
    """双向 LSTM 文本分类器，循环单元完全手写。"""

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = _trainable_embedding(embedding_matrix)
        embedding_dim = embedding_matrix.size(1)
        self.lstm = ManualBidirectionalRecurrentEncoder(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            cell_type="lstm",
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """只使用 LSTM 的 hidden state，不使用 cell state 做分类。"""
        embedded = self.embedding(inputs)
        _, (hidden, _) = self.lstm(embedded, lengths)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        features = torch.cat([forward_hidden, backward_hidden], dim=1)
        return self.fc(self.dropout(features))


class AdditiveAttention(nn.Module):
    """对序列输出做带掩码的加性注意力池化。"""

    def __init__(self, input_dim: int, attention_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = sequence.size()
        positions = torch.arange(seq_len, device=sequence.device).unsqueeze(0).expand(batch_size, seq_len)
        mask = positions < lengths.clamp(min=1).unsqueeze(1)

        scores = self.score(torch.tanh(self.proj(sequence))).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), sequence).squeeze(1)
        return context, weights


class BiLSTMAttentionClassifier(nn.Module):
    """在双向 LSTM 输出上做注意力池化的文本分类器。"""

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        attention_dim: int = 128,
    ):
        super().__init__()
        self.embedding = _trainable_embedding(embedding_matrix)
        embedding_dim = embedding_matrix.size(1)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = AdditiveAttention(hidden_dim * 2, attention_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """让模型从整条序列中学习更重要的情感触发词。"""
        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(
            embedded,
            lengths.clamp(min=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, _ = self.lstm(packed)
        sequence_outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=inputs.size(1),
        )
        context, _ = self.attention(sequence_outputs, lengths)
        return self.fc(self.dropout(context))


def initialize_model(model: nn.Module) -> None:
    """按层类型选择更合适的初始化策略。"""
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 预训练词向量已经有意义，不再用随机初始化覆盖。
        if "embedding" in name:
            continue
        if (name.endswith("norm.weight") or ("norm" in name and name.endswith("weight"))):
            nn.init.ones_(param)
        elif (name.endswith("norm.bias") or ("norm" in name and name.endswith("bias"))):
            nn.init.zeros_(param)
        elif "weight_hh" in name:
            nn.init.orthogonal_(param)
        elif "weight_ih" in name:
            nn.init.xavier_uniform_(param)
        elif "conv" in name and "weight" in name:
            nn.init.kaiming_uniform_(param, nonlinearity="relu")
        elif "weight" in name and param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)


def count_parameters(model: nn.Module) -> int:
    """统计需要训练的参数量，用于实验结果表展示。"""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
