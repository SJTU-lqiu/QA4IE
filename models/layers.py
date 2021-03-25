import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


NEG_INF = -1e30
POS_INF = 1e30


class CharCNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.1):
        super(CharCNNLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # self.conv = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Conv1d(in_channels, out_channels, kernel_size),
        #     nn.ReLU(),
        #     nn.MaxPool1d(1)
        # )

    def forward(self, x):
        conv_out = self.relu(self.conv(self.dropout(x))).max(dim=-1).values
        return conv_out


class CharCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, dropout=0.1):
        super(CharCNN, self).__init__()
        self.conv_layers = nn.ModuleList([
            CharCNNLayer(in_channels, out_c, kernel_size, dropout)
        for out_c, kernel_size in zip(out_channels, kernel_sizes)])
    
    def forward(self, x):
        """
        params:
            x: [N, L, W, di]
        """
        N, L, W, di = x.size()
        x = x.reshape(N*L, W, di).transpose(-1, -2)  # [N*L, di, W]
        conv_out = torch.cat([
            conv_layer(x) for conv_layer in self.conv_layers
        ], dim=1)  # [N*L, do]
        return conv_out.reshape(N, L, -1)


class Embedding(nn.Module):
    def __init__(self, char_vocab_size, char_emb_size, word_emb, out_channels, kernel_sizes, dropout=0.1):
        super(Embedding, self).__init__()

        # char emb
        self.char_emb = nn.Embedding(char_vocab_size, char_emb_size, padding_idx=0)

        # char cnn
        self.char_cnn = CharCNN(char_emb_size, out_channels, kernel_sizes, dropout)

        # word emb
        self.word_emb = nn.Embedding.from_pretrained(torch.from_numpy(word_emb), freeze=True, padding_idx=0)

    def forward(self, char_context, char_query, context, query):
        """
        params:
            char_context: [N, Lc, W]
            char_query: [N, Lq, W]
            context: [N, Lc]
            query: [N, Lq]
        """
        # [N, Lc, W, dc]  [N, Lq, W, dc]
        c_ctx, c_q = self.char_emb(char_context), self.char_emb(char_query)
        # print(self.char_cnn.conv_layers[0].conv.weight)
        # exit(0)
        # print(c_ctx.mean(dim=-1))
        # print(c_q.mean(dim=-1))
        # print(c_ctx.min(), c_q.min(), c_ctx.max(), c_q.max())
        c_ctx, c_q = self.char_cnn(c_ctx), self.char_cnn(c_q)  # [N, Lc, dco]  [N, Lq, dco]
        # print(c_ctx.min(), c_q.min(), c_ctx.max(), c_q.max())
        # exit(0)
        # print(c_ctx.mean(dim=-1))
        # print(c_q.mean(dim=-1))
        # [N, Lc, d]  [N, Lq, d]
        ctx, q = self.word_emb(context), self.word_emb(query)
        # print(ctx.mean(dim=-1))
        # print(q.mean(dim=-1))
        # print()
        # exit(0)
        ctx, q = torch.cat([c_ctx, ctx], dim=-1), torch.cat([c_q, q], dim=-1)
        
        return ctx, q


class HighwayLayer(nn.Module):
    def __init__(self, size):
        super(HighwayLayer, self).__init__()
        self.trans = nn.Sequential(
            nn.Linear(size, size),
            nn.ReLU()
        )
        self.gate = nn.Sequential(
            nn.Linear(size, size),
            nn.Sigmoid()
        )

    def forward(self, x):
        gate, trans = self.gate(x), self.trans(x)
        return gate * trans + (1 - gate) * x


class HighwayNetwork(nn.Module):
    def __init__(self, num_layers, size):
        super(HighwayNetwork, self).__init__()
        self.highway_layers = nn.ModuleList([
            HighwayLayer(size) for _ in range(num_layers)
        ])

    def forward(self, x):
        for highway_layer in self.highway_layers:
            x = highway_layer(x)
        return x


class RNNPackWrapper(nn.Module):
    def __init__(self, rnn):
        super(RNNPackWrapper, self).__init__()
        self.rnn = rnn

    def forward(self, x, x_masks):
        """
        params:
            x: [N, L]
            x_masks: [N, L]
        """
        L = x.size(1)
        x_len = x_masks.sum(dim=1)
        lengths, sort_idx = x_len.sort(0, descending=True)
        x = pack_padded_sequence(x[sort_idx], lengths.cpu(), batch_first=True)

        out, final = self.rnn(x)

        out, _ = pad_packed_sequence(out, batch_first=True, total_length=L)
        _, unsort_idx = sort_idx.sort(0)
        out = out[unsort_idx]

        return out, final


class BiAttentionFlow(nn.Module):
    def __init__(self, dim):
        super(BiAttentionFlow, self).__init__()
        self.linear = nn.Linear(6 * dim, 1)

    def forward(self, ctx, q, ctx_mask, q_mask):
        Lc, Lq = ctx.size(1), q.size(1)
        ctx_aug, q_aug = ctx[..., None, :].repeat(1, 1, Lq, 1), q[:, None, ...].repeat(1, Lc, 1, 1)  # [N, Lc, Lq, 2d]  [N, Lc, Lq, 2d]
        ctx_q_mask = ctx_mask[..., None] & q_mask[:, None, :]  # [N, Lc, Lq]
        ctx_q_feature = torch.cat([
            ctx_aug, q_aug, ctx_aug * q_aug
        ], dim=-1)  # [N, Lc, Lq, 6d]
        # similarity matrix
        S = self.linear(ctx_q_feature).squeeze(-1) + (1 - ctx_q_mask.float()) * NEG_INF  # [N, Lc, Lq]

        # c2q attention
        c2q_attention = S.softmax(dim=-1)
        # print(c2q_attention[0])
        # print(c2q_attention[1])
        u = (c2q_attention.unsqueeze(-1) * q_aug).sum(dim=2)  # [N, Lc, 2d]
        # q2c attention
        q2c_attention = S.max(dim=-1, keepdim=True).values.softmax(dim=1)  # [N, Lc, 1]
        h = (ctx * q2c_attention).sum(dim=1, keepdim=True).repeat(1, Lc, 1)  # [N, Lc, 2d]

        return torch.cat([
            ctx, u, ctx * u, ctx * h
        ], dim=-1)


class SelfMatching(nn.Module):
    def __init__(self, dim, dropout):
        super(SelfMatching, self).__init__()
        self.linear1 = nn.Linear(2 * dim, dim)
        self.linear2 = nn.Linear(2 * dim, dim)
        self.tanh = nn.Tanh()
        self.linear3 = nn.Linear(dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(4 * dim, dim, batch_first=True, bidirectional=True)

    def forward(self, x, x_mask):
        """
        params:
            x: [N, Lc, 2d]
            x_mask: [N, Lc]
        """
        aug_x = self.linear1(x).unsqueeze(1) + self.linear2(x).unsqueeze(2)  # [N, Lc, Lc, d]
        S = self.linear3(self.tanh(aug_x)).squeeze(-1)  # [N, Lc, Lc]
        # add mask
        S = S + (1 - x_mask.unsqueeze(1).float()) * NEG_INF
        S.softmax(dim=-1)
        C = (S.unsqueeze(-1) * x.unsqueeze(1)).sum(dim=2)  # [N, Lc, 2d]
        C_aug = torch.cat([x, C], dim=-1)  # [N, Lc, 4d]
        out, final_state = self.lstm(self.dropout(C_aug))  # [N, Lc, 2d]

        h, c = final_state
        h, c = h.transpose(0, 1).reshape(h.size(1), -1), c.transpose(0, 1).reshape(c.size(1), -1)

        return out, (h, c)


class BiDAF(nn.Module):
    def __init__(self, config):
        super(BiDAF, self).__init__()
        self.config = config

        # 1. Character Embedding Layer
        self.char_emb = nn.Embedding(config.char_vocab_size, config.char_emb_size, padding_idx=0)
        nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)

        self.char_conv = nn.Sequential(
            nn.Conv2d(1, int(config.out_channels), (config.char_emb_size, int(config.kernel_sizes))),
            # nn.Conv2d(1, args.char_channel_size, (args.char_dim, args.char_channel_width)),
            nn.ReLU()
            )

        # 2. Word Embedding Layer
        # initialize word embedding with GloVe
        self.word_emb = nn.Embedding.from_pretrained(config.word_emb, freeze=True)

        # highway network
        assert config.hidden_size * 2 == (config.char_out_size + config.word_emb_size)
        for i in range(2):
            setattr(self, 'highway_linear{}'.format(i),
                    nn.Sequential(nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
                                  nn.ReLU()))
            setattr(self, 'highway_gate{}'.format(i),
                    nn.Sequential(nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
                                  nn.Sigmoid()))

        # 3. Contextual Embedding Layer
        self.context_LSTM = nn.LSTM(input_size=config.hidden_size * 2,
                                 hidden_size=config.hidden_size,
                                 bidirectional=True,
                                 batch_first=True,
                                 dropout=config.dropout)

        # 4. Attention Flow Layer
        self.att_weight_c = nn.Linear(config.hidden_size * 2, 1)
        self.att_weight_q = nn.Linear(config.hidden_size * 2, 1)
        self.att_weight_cq = nn.Linear(config.hidden_size * 2, 1)

        # 5. Modeling Layer
        self.modeling_LSTM1 = nn.LSTM(input_size=config.hidden_size * 8,
                                   hidden_size=config.hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=config.dropout)

        self.modeling_LSTM2 = nn.LSTM(input_size=config.hidden_size * 2,
                                   hidden_size=config.hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=config.dropout)

        # 6. Output Layer
        self.p1_weight_g = nn.Linear(config.hidden_size * 8, 1, dropout=config.dropout)
        self.p1_weight_m = nn.Linear(config.hidden_size * 2, 1, dropout=config.dropout)
        self.p2_weight_g = nn.Linear(config.hidden_size * 8, 1, dropout=config.dropout)
        self.p2_weight_m = nn.Linear(config.hidden_size * 2, 1, dropout=config.dropout)

        self.output_LSTM = nn.LSTM(input_size=config.hidden_size * 2,
                                hidden_size=config.hidden_size,
                                bidirectional=True,
                                batch_first=True,
                                dropout=config.dropout)

        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, cx, cq, x, q, c_mask, q_mask):
        # TODO: More memory-efficient architecture
        def char_emb_layer(x):
            """
            :param x: (batch, seq_len, word_len)
            :return: (batch, seq_len, char_channel_size)
            """
            batch_size = x.size(0)
            # (batch, seq_len, word_len, char_dim)
            x = self.dropout(self.char_emb(x))
            # (batch， seq_len, char_dim, word_len)
            x = x.transpose(2, 3)
            # (batch * seq_len, 1, char_dim, word_len)
            x = x.view(-1, self.config.char_emb_size, x.size(3)).unsqueeze(1)
            # (batch * seq_len, char_channel_size, 1, conv_len) -> (batch * seq_len, char_channel_size, conv_len)
            x = self.char_conv(x).squeeze()
            # (batch * seq_len, char_channel_size, 1) -> (batch * seq_len, char_channel_size)
            x = F.max_pool1d(x, x.size(2)).squeeze()
            # (batch, seq_len, char_channel_size)
            x = x.view(batch_size, -1, int(self.config.out_channels))

            return x

        def highway_network(x1, x2):
            """
            :param x1: (batch, seq_len, char_channel_size)
            :param x2: (batch, seq_len, word_dim)
            :return: (batch, seq_len, hidden_size * 2)
            """
            # (batch, seq_len, char_channel_size + word_dim)
            x = torch.cat([x1, x2], dim=-1)
            for i in range(2):
                h = getattr(self, 'highway_linear{}'.format(i))(x)
                g = getattr(self, 'highway_gate{}'.format(i))(x)
                x = g * h + (1 - g) * x
            # (batch, seq_len, hidden_size * 2)
            return x

        def att_flow_layer(c, q):
            """
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, q_len)
            """
            c_len = c.size(1)
            q_len = q.size(1)

            # (batch, c_len, q_len, hidden_size * 2)
            #c_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            #q_tiled = q.unsqueeze(1).expand(-1, c_len, -1, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            #cq_tiled = c_tiled * q_tiled
            #cq_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1) * q.unsqueeze(1).expand(-1, c_len, -1, -1)

            cq = []
            for i in range(q_len):
                #(batch, 1, hidden_size * 2)
                qi = q.select(1, i).unsqueeze(1)
                #(batch, c_len, 1)
                ci = self.att_weight_cq(c * qi).squeeze()
                cq.append(ci)
            # (batch, c_len, q_len)
            cq = torch.stack(cq, dim=-1)

            # (batch, c_len, q_len)
            s = self.att_weight_c(c).expand(-1, -1, q_len) + \
                self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
                cq

            # (batch, c_len, q_len)
            a = F.softmax(s, dim=2)
            # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
            c2q_att = torch.bmm(a, q)
            # (batch, 1, c_len)
            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
            q2c_att = torch.bmm(b, c).squeeze()
            # (batch, c_len, hidden_size * 2) (tiled)
            q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
            # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

            # (batch, c_len, hidden_size * 8)
            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
            return x

        def output_layer(g, m, l):
            """
            :param g: (batch, c_len, hidden_size * 8)
            :param m: (batch, c_len ,hidden_size * 2)
            :return: p1: (batch, c_len), p2: (batch, c_len)
            """
            # (batch, c_len)
            p1 = (self.p1_weight_g(g) + self.p1_weight_m(m)).squeeze()
            # (batch, c_len, hidden_size * 2)
            m2 = self.output_LSTM((m, l))[0]
            # (batch, c_len)
            p2 = (self.p2_weight_g(g) + self.p2_weight_m(m2)).squeeze()

            return p1, p2

        # 1. Character Embedding Layer
        c_char = char_emb_layer(cx)
        q_char = char_emb_layer(cq)
        # 2. Word Embedding Layer
        c_word = self.word_emb(batch.c_word[0])
        q_word = self.word_emb(batch.q_word[0])
        c_lens = batch.c_word[1]
        q_lens = batch.q_word[1]

        # Highway network
        c = highway_network(c_char, c_word)
        q = highway_network(q_char, q_word)
        # 3. Contextual Embedding Layer
        c = self.context_LSTM((c, c_lens))[0]
        q = self.context_LSTM((q, q_lens))[0]
        # 4. Attention Flow Layer
        g = att_flow_layer(c, q)
        # 5. Modeling Layer
        m = self.modeling_LSTM2((self.modeling_LSTM1((g, c_lens))[0], c_lens))[0]
        # 6. Output Layer
        p1, p2 = output_layer(g, m, c_lens)

        # (batch, c_len), (batch, c_len)
        return p1, p2


if __name__ == "__main__":
    conv = nn.Conv1d(32, 64, 5)
    in_ = torch.randn(4, 32, 12)
    print(conv(in_).repeat(2, 1, 2).size())
    lstm = nn.LSTM(32, 32, batch_first=True, bidirectional=True, num_layers=2)
    in_ = torch.randn(4, 10, 32)
    out, (h, c) = lstm(in_)
    print(out.size(), h.size(), c.size())