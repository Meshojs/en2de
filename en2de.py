from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer
from torch import nn
from torch._subclasses.fake_impls import data_dep
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    # data
    vocab_size: int = 16000
    max_len: int = 50
    batch_size: int = 128
    # model
    emb_dim: int = 256
    hidden_size: int = 512
    n_layers: int = 2
    dropout: float = 0.3
    # training
    epochs: int = 25
    lr: float = 1e-3
    betas: tuple = (0.9, 0.999)
    label_smoothing: float = 0.1
    clip_norm: float = 1.0
    # scheduler
    warmup_ratio: float = 0.05
    start_factor: float = 0.01
    eta_min: float = 1e-5


cfg = Config()
"""Tokenizer"""
tok = Tokenizer.from_file("./en2de.json")
"""Apply For Dataset"""
ds: Any = load_dataset("bentrevett/multi30k")
en = ds["train"]["en"]
de = ds["train"]["de"]
en_val = ds["validation"]["en"]
de_val = ds["validation"]["de"]
pairs = list(zip(en, de))
val_pairs = list(zip(en_val, de_val))
BOS, PAD, EOS = (
    Tokenizer.token_to_id(tok, "<bos>"),
    Tokenizer.token_to_id(tok, "<pad>"),
    Tokenizer.token_to_id(tok, "<eos>"),
)


class en2de_dataset(Dataset):
    def __init__(self, pairs, tokenizer):
        self.data = []
        for src, tgt in pairs:
            s = torch.tensor(tokenizer.encode(src).ids + [EOS])
            t = torch.tensor([BOS] + tokenizer.encode(tgt).ids + [EOS])
            self.data.append((s, t))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate_fn(batch):
    src, tgt = zip(*batch)
    src = pad_sequence(src, batch_first=True, padding_value=PAD)
    tgt = pad_sequence(tgt, batch_first=True, padding_value=PAD)
    return src, tgt


dataset = en2de_dataset(pairs, tok)
loader = DataLoader(dataset, shuffle=True, batch_size=cfg.batch_size, collate_fn=collate_fn)
val_dataset = en2de_dataset(pairs, tok)
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=cfg.batch_size, collate_fn=collate_fn)


class Model(nn.Module):
    def __init__(self, n_layers, hidden_size, vocab_size, emb_dim, pad_id=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.enc = nn.LSTM(emb_dim, hidden_size, n_layers, batch_first=True)
        self.dec = nn.LSTM(emb_dim, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, src, dec_in):
        _, (h, c) = self.enc(self.emb(src))  # encode
        out, _ = self.dec(self.emb(dec_in), (h, c))  # decode, init with h, c
        return self.fc(out)  # (B, tgt_len, vocab)


model = Model(cfg.n_layers, cfg.hidden_size, cfg.vocab_size, cfg.emb_dim).to(device)
opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=cfg.betas)
crit = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=cfg.label_smoothing)
total = cfg.epochs * len(loader)
warm = int(cfg.warmup_ratio * total)
sched = SequentialLR(
    opt,
    schedulers=[
        LinearLR(opt, start_factor=0.01, total_iters=warm),
        CosineAnnealingLR(opt, T_max=total - warm, eta_min=1e-5),
    ],
    milestones=[warm],
)

def run(loader, train):
    model.train(train)
    total_loss = 0
    with torch.set_grad_enabled(train):
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, tgt[:, :-1])
            loss = crit(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()  # per batch
            total_loss += loss.item()
    return total_loss / len(loader)


train_l, val_l, best = [], [], float("inf")

for ep in range(cfg.epochs):
    train_l.append(run(loader, True))
    val_l.append(run(val_loader, False))
    print(f"ep {ep + 1} | train {train_l[-1]:.3f} | val {val_l[-1]:.3f}")

    if val_l[-1] < best:
        best = val_l[-1]
        torch.save(model.state_dict(), "best.pt")

plt.plot(train_l, label="train")
plt.plot(val_l, label="val")
plt.legend()
plt.show()
