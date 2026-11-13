# en2de - Seq2Seq
### English → German
An encoder–decoder translation model implemented from scratch in PyTorch and trained on Multi30k.

## Overview
| | |
|---|---|
| Architecture | 2-layer LSTM encoder–decoder (hidden 512, embedding 256) |
| Data | Multi30k (~29k EN–DE caption pairs) |
| Tokenizer | Custom BPE - 16k size | 
| Optimization | Adam (lr 1e-3), linear warmup + cosine decay, label smoothing 0.1 |
| Decoding | Greedy |



<hr>
<img width="640" height="520" alt="en2de---By-Mashary-❤️" src="https://github.com/user-attachments/assets/5ca78119-80eb-49d4-a8bb-7052de50281f" />
<hr>

<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/e4678325-300c-4cbb-b971-9fdd3ee0b0f8" />

<hr>

## Performance
- Produces fluent, largely accurate translations on in-domain, caption-style sentences.
- Does not generalize to out-of-domain text (news, questions, first-person speech).
- BLEU: not yet evaluated.

## Limitations
- No attention mechanism; the encoder state is a fixed-size bottleneck.
- Small, single-domain training set.
