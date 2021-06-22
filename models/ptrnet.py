import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import NEG_INF


class PointerNetwork(nn.Module):
    def __init__(self, dim, attn_dim, max_decode_len):
        super(PointerNetwork, self).__init__()
        self.max_decode_len = max_decode_len
        self.decode_cell = nn.LSTMCell(2 * dim, 2 * dim)
        self.linear1 = nn.Linear(2 * dim, attn_dim)
        self.linear2 = nn.Linear(2 * dim, attn_dim)
        self.tanh = nn.Tanh()
        self.linear3 = nn.Linear(attn_dim, 1)
        self.eos_vector = nn.Parameter(torch.empty([1, 2 * dim]))
        torch.nn.init.xavier_normal_(self.eos_vector)
        self.enc_size = 2 * dim

    def forward(self, encoder_outputs, encoder_final_state, mask):
        """
        params:
            encoder_outputs: [N, L, 2d]
            encoder_final_state: ([N, 2d], [N, 2d])
            mask: [N, L]
        """
        lens = mask.sum(dim=1)
        N, L, size = encoder_outputs.size()
        cur_input = torch.zeros(N, size).to(encoder_outputs)
        cur_state = encoder_final_state
        # max_length = min(int(mask.sum(dim=1).max()), self.max_decode_len)

        encoder_outputs = torch.stack([
            # [L+1, 2d]
            torch.cat([
                encoder_outputs[i, :lens[i]],
                self.eos_vector,
                encoder_outputs[i, lens[i]:]
            ], dim=0) for i in range(N)
        ], dim=0)

        outputs = []
        for _ in range(self.max_decode_len):
            h, c = self.decode_cell(cur_input, cur_state)
            attn = self.linear3(
                self.tanh(
                    self.linear1(h).unsqueeze(1) + self.linear2(encoder_outputs)
                )
            ).squeeze(-1)  # [N, L]
            attn = attn + (1 - torch.cat([torch.ones(N, 1).to(mask), mask], dim=1).float()) * NEG_INF
            attended_input = (attn.softmax(dim=-1).unsqueeze(-1) * encoder_outputs).sum(dim=1)
            cur_input = attended_input
            cur_state = (h, c)
            outputs.append(attn)

        return torch.stack(outputs, dim=1)  # [N, La, Lc]
