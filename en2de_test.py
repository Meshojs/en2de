# test.py
import torch
from torch import nn
from tokenizers import Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

tok = Tokenizer.from_file("./en2de.json")
BOS, EOS, PAD = tok.token_to_id("<bos>"), tok.token_to_id("<eos>"), tok.token_to_id("<pad>")

class Model(nn.Module):
    def __init__(self, n_layers, hidden_size, vocab_size, emb_dim, pad_id=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.enc = nn.LSTM(emb_dim, hidden_size, n_layers, batch_first=True)
        self.dec = nn.LSTM(emb_dim, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

model = Model(2, 512, tok.get_vocab_size(), 256, PAD).to(device)
model.load_state_dict(torch.load("best.pt", map_location=device))
model.eval()

@torch.no_grad()
def translate(text, max_len=50):
    src = torch.tensor([tok.encode(text).ids + [EOS]]).to(device)
    _, (h, c) = model.enc(model.emb(src))
    x = torch.tensor([[BOS]]).to(device)
    out = []
    for _ in range(max_len):
        o, (h, c) = model.dec(model.emb(x), (h, c))
        x = model.fc(o[:, -1]).argmax(-1, keepdim=True)
        if x.item() == EOS:
            break
        out.append(x.item())
    return tok.decode(out)




sentences = [
    "A man is cooking in the kitchen",
    # "Two dogs are playing in the snow.",
    # "A woman is cooking in the kitchen.",
    # "Children are playing football in the park.",
    # "A girl in a red dress is walking down the street.",
    # "A group of people are standing in front of a building.",
]

for s in sentences:
    print("EN:", s)
    print("DE:", translate(s).replace(" .", "."))
    print()
