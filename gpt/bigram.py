# try bigram
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(1337)

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('gpt/input.txt', 'r') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)

# create mapping from char to intrgers
stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for ch, i in stoi.items() }

encode = lambda s: [ stoi[c] for c in s ]
decode = lambda n: ''.join([itos[i] for i in n])

# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# data loader
def get_batch(split):
    # generate small batches
    data = train_data if split == 'train' else val_data    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class BigramLanquageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None):
        
        # (B,T) integers
        logits = self.token_embedding_table(idx)  # (B,T,C)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices
        for _ in range(max_new_tokens):
            # predict
            logits, loss = self(idx)  # (B, T, C)
            # get last time step
            logits = logits[:,-1,:]  # (B, C)
            # turn to probabilities
            probs = F.softmax(logits, dim=1)  # (B, C)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append ampled to idx
            idx = torch.concat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

model = BigramLanquageModel(vocab_size)
m = model.to(device)

# create optimizer
optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # eval model
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample from the model
    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
