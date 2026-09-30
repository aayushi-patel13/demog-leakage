"""Momentum SGD that behaves like DyNet's MomentumSGDTrainer.

Two details of the original DyNet setup matter for speed and faithfulness:

* Lookup (embedding) parameters get *sparse* updates: only the rows of words in
  the current mini-batch are touched, including their momentum. With a 100k+
  word vocabulary, a dense update of the whole 300-d embedding table every step
  makes CPU training about 4x slower, so we reproduce the sparse behaviour.
* Gradients are clipped to a global L2 norm of 5 (DyNet's default).
"""
import torch


class DyNetStyleSGD:
    def __init__(self, model, lr=0.01, momentum=0.9, clip=5.0):
        self.lr, self.mom, self.clip = lr, momentum, clip
        self.sparse = [m.weight for m in model.modules()
                       if isinstance(m, torch.nn.Embedding) and m.sparse]
        sparse_ids = {id(p) for p in self.sparse}
        self.dense = [p for p in model.parameters() if id(p) not in sparse_ids]
        self.opt = torch.optim.SGD(self.dense, lr=lr, momentum=momentum)
        self.bufs = {id(p): torch.zeros_like(p) for p in self.sparse}

    def zero_grad(self):
        self.opt.zero_grad()
        for p in self.sparse:
            p.grad = None

    @torch.no_grad()
    def step(self):
        coalesced = []
        sq = torch.zeros((), device=self.bufs[id(self.sparse[0])].device if self.sparse else "cpu")
        for p in self.dense:
            if p.grad is not None:
                sq = sq + p.grad.pow(2).sum().to(sq.device)
        for p in self.sparse:
            if p.grad is None:
                coalesced.append(None)
                continue
            g = p.grad.coalesce()
            coalesced.append(g)
            sq = sq + g.values().pow(2).sum()
        norm = sq.sqrt().item()
        scale = min(1.0, self.clip / (norm + 1e-6)) if self.clip else 1.0
        if scale < 1.0:
            for p in self.dense:
                if p.grad is not None:
                    p.grad.mul_(scale)
        self.opt.step()
        for p, g in zip(self.sparse, coalesced):
            if g is None:
                continue
            idx = g.indices()[0]
            vals = g.values() * scale
            buf = self.bufs[id(p)]
            buf[idx] = self.mom * buf[idx] + vals
            p[idx] -= self.lr * buf[idx]
        return norm
