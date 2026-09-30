"""PyTorch re-implementation of AdvNN / AttackerNN from Elazar & Goldberg (2018).

Sizes follow the original trainer.py defaults: 300-d word embeddings, a
one-layer LSTM with 300 hidden units whose final state is the representation
h, and 300-unit tanh MLP heads with dropout 0.2. The adversary sits behind a
gradient reversal layer (Ganin & Lempitsky, 2015), dy.flip_gradient in DyNet.
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

EMB = 300
HID = 300


class GradReverse(torch.autograd.Function):
    """Identity on the forward pass; multiplies the gradient by -lambda backward."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb=EMB, hid=HID):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0, sparse=True)  # sparse rows, like DyNet lookups
        self.lstm = nn.LSTM(emb, hid, batch_first=True)

    def forward(self, x, lens):
        packed = pack_padded_sequence(self.emb(x), lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return h[-1]  # final hidden state = sentence representation


class MLP(nn.Module):
    """affine -> dropout -> tanh -> affine, as task_mlp / adv_mlp / AttackerNN."""

    def __init__(self, d_in=HID, d_hid=HID, n_out=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.Dropout(dropout), nn.Tanh(),
                                 nn.Linear(d_hid, n_out))

    def forward(self, h):
        return self.net(h)


class AdvModel(nn.Module):
    """Encoder + main-task head, plus (optionally) adversaries behind a GRL."""

    def __init__(self, vocab_size, n_adv=1, adv_hid=HID):
        super().__init__()
        self.encoder = Encoder(vocab_size)
        self.task = MLP()
        self.advs = nn.ModuleList([MLP(d_hid=adv_hid) for _ in range(n_adv)])

    def forward(self, x, lens, lambd=0.0):
        h = self.encoder(x, lens)
        task_logits = self.task(h)
        adv_logits = [a(GradReverse.apply(h, lambd)) for a in self.advs] if lambd > 0 else []
        return h, task_logits, adv_logits
