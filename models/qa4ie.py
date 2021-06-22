import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Embedding, HighwayNetwork, BiAttentionFlow, NEG_INF, SelfMatching, RNNPackWrapper
from .ptrnet import PointerNetwork


class QA4IEQA(nn.Module):
    def __init__(self, config):
        super(QA4IEQA, self).__init__()
        out_channels = list(map(int, config.out_channels.split(',')))
        kernel_sizes = list(map(int, config.kernel_sizes.split(',')))
        assert sum(out_channels) == config.char_out_size
        self.emb = Embedding(
            config.char_vocab_size, config.char_emb_size, config.word_emb, 
            out_channels, kernel_sizes, config.dropout
        )
        d = config.word_emb.shape[1] + config.char_out_size
        self.dropout = nn.Dropout(config.dropout)
        self.highway = HighwayNetwork(config.num_highway_layers, d)
        self.encode_lstm = RNNPackWrapper(nn.LSTM(d, config.hidden_size, batch_first=True, bidirectional=True))
        self.biattention = BiAttentionFlow(config.hidden_size)
        self.model_lstm = RNNPackWrapper(nn.LSTM(8 * config.hidden_size, config.hidden_size, num_layers=config.num_modeling_layers,
            batch_first=True, bidirectional=True, dropout=config.dropout))
        self.matching = SelfMatching(config.hidden_size, config.dropout)
        self.ptrnet = PointerNetwork(config.hidden_size, config.attn_proj_size, config.max_decode_length)
        self.loss_fnt = nn.CrossEntropyLoss(reduction='none')

    def forward(self, char_context, char_query, context, query, context_mask, query_mask, labels=None, labels_mask=None):
        """
        params:
            char_context: [N, Lc, W]
            char_query: [N, Lq, W]
            context: [N, Lc]
            query: [N, Lq]
            context_mask: [N, Lc]
            query_mask: [N, Lq]
        """
        # embedding
        ctx, q = self.emb(char_context, char_query, context, query)  # [N, Lc, d]  # [N, Lq, d]
        # highway
        ctx, q = self.highway(ctx), self.dropout(self.highway(q))
        # contextual embedding
        ctx, _ = self.encode_lstm(ctx, context_mask)  # [N, Lc, 2d]
        q, _ = self.encode_lstm(q, query_mask)  # [N, Lq, 2d]
        # attention flow
        q_aware_ctx = self.dropout(self.biattention(ctx, q, context_mask, query_mask))
        # modeling layer
        g, _ = self.model_lstm(q_aware_ctx, context_mask)
        # self matching layer
        g, encoder_final_state = self.matching(g, context_mask)
        encoder_outputs = g * context_mask.unsqueeze(-1).float()
        # pre pointer network
        output = self.ptrnet(encoder_outputs, encoder_final_state, context_mask)

        if labels is None:
            return output

        N, La, Lc = output.size()
        assert labels.max() < Lc, (labels, labels.max(), Lc)
        loss = (
            self.loss_fnt(
                output.reshape(N * La, -1), labels.reshape(-1)
            ) * labels_mask.float().reshape(-1)
        ).sum() / labels_mask.sum()

        return loss, output


class GroupEncoder(nn.Module):
    def __init__(self, config):
        super(GroupEncoder, self).__init__()
        out_channels = list(map(int, config.out_channels.split(',')))
        kernel_sizes = list(map(int, config.kernel_sizes.split(',')))
        assert sum(out_channels) == config.char_out_size
        self.emb = Embedding(
            config.char_vocab_size, config.char_emb_size, config.word_emb,
            out_channels, kernel_sizes, config.dropout
        )
        d = config.word_emb.shape[1] + config.char_out_size
        self.dropout = nn.Dropout(config.dropout)
        self.highway = HighwayNetwork(config.num_highway_layers, d)
        self.encode_lstm = RNNPackWrapper(nn.LSTM(d, config.hidden_size, batch_first=True, bidirectional=True))
        self.biattention = BiAttentionFlow(config.hidden_size)
        self.model_lstm = RNNPackWrapper(nn.LSTM(
            8 * config.hidden_size, config.hidden_size, batch_first=True, bidirectional=True))
        self.linear1 = nn.Linear(2 * config.hidden_size, config.attn_proj_size)
        self.linear2 = nn.Linear(config.attn_proj_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, char_context, char_query, context, query, context_mask, query_mask):
        ctx, q = self.emb(char_context, char_query, context, query)
        ctx, q = self.highway(ctx), self.dropout(self.highway(q))
        ctx, _ = self.encode_lstm(ctx, context_mask)

        q, _ = self.encode_lstm(q, query_mask)
        q_aware_ctx = self.dropout(self.biattention(ctx, q, context_mask, query_mask))
        g, _ = self.model_lstm(q_aware_ctx, context_mask)  # [N, Lc, 2d]
        attn = self.linear2(self.tanh(self.linear1(g)))  # [N, Lc, 1]
        attn = (attn + (1 - context_mask.float().unsqueeze(-1)) * NEG_INF).softmax(dim=1)
        attended_output = (attn * g).sum(dim=1)  # [N, 2d]

        return attended_output


class QA4IESS(nn.Module):
    def __init__(self, config):
        super(QA4IESS, self).__init__()
        self.encoder = GroupEncoder(config)
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(2 * config.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # SS dataset neg / pos ~= 42
        # self.loss_fnt = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(42.).to(0))
        self.loss_fnt = nn.BCEWithLogitsLoss()

    def forward(self, char_context, char_query, context, query, context_mask, query_mask, labels=None):
        attended_output = self.encoder(char_context, char_query, context, query, context_mask, query_mask)
        logits = self.classifier(attended_output).squeeze(-1)  # [N]
        if labels is None:
            return logits
        loss = self.loss_fnt(logits, labels)
        return (loss, logits)


class QA4IEAT(nn.Module):
    def __init__(self, config):
        super(QA4IEAT, self).__init__()
        self.encoder = GroupEncoder(config)
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(2 * config.hidden_size + config.ss_feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.loss_fnt = nn.BCEWithLogitsLoss()

    def forward(self, char_context, char_query, context, query, context_mask, query_mask, ss_feature, labels=None):
        attended_output = self.encoder(char_context, char_query, context, query, context_mask, query_mask)
        logits = self.classifier(
            torch.cat([attended_output, ss_feature], dim=-1)
        ).squeeze(-1)
        if labels is None:
            return logits
        loss = self.loss_fnt(logits, labels.float())
        return (loss, logits)


if __name__ == "__main__":
    from config import get_args
    config = get_args()
    config.char_vocab_size = 100
    config.word_vocab_size = 100
    model = QA4IEQA(config)

