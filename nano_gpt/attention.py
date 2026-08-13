import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# %matplotlib inline

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using device: {device}')
eval_iters = 200
n_head = 6
n_layer = 6
n_embd = 384 
dropout = 0.2
# ----------------

torch.manual_seed(1337)

text = open('input.txt', 'r', encoding='utf-8').read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# mapping between char and integer for each unique character
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # output list of all integers
decode = lambda a: ''.join([itos[i] for i in a]) # output string

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  return x.to(device), y.to(device)

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

class MultiHeadAttention(nn.Module):
  def __init__(self):
    super().__init__()
    self.key = nn.Linear(n_embd, n_embd, bias=False)
    self.query = nn.Linear(n_embd, n_embd, bias=False)
    self.value = nn.Linear(n_embd, n_embd, bias=False)
    self.proj = nn.Linear(n_embd, n_embd)
    self.register_buffer('bias', torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    B, T, C = x.size()

    # calculate key, query, value by splitting the embedding into heads 
    k = self.key(x).view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, n_head, T, head_size)
    q = self.query(x).view(B, T, n_head, C // n_head).transpose(1, 2) # (B, n_head, T, head_size)
    v = self.value(x).view(B, T, n_head, C // n_head).transpose(1, 2) # (B, n_head, T, head_size)
    
    wei = q @ k.transpose(-2, -1) * C**-0.5
    wei = wei.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    out = wei @ v

    # rearrange the output back to the original shape
    out = out.transpose(1, 2).contiguous().view(B, T, C)
    return self.proj(out)

class FeedForward(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd),
      nn.ReLU(),
      nn.Linear(4 * n_embd, n_embd),
      nn.Dropout(dropout),
    )
  def forward(self, x):
    return self.net(x)

class Block(nn.Module):

  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention()
    self.ff = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)
  
  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ff(self.ln2(x))
    return x

class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)],)
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)
  
  def forward(self, idx, targets=None):
    
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx) # (B,T,C)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
    x = tok_emb + pos_emb # (B,T,C)
    x = self.blocks(x)
    logits = self.lm_head(x) # (B,T,vocab_size)
    
    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def genereate(self, idx, max_new_tokens):

    for _ in range(max_new_tokens):
      # crop context window to block_size tokens
      idx_cond = idx[:, -block_size:]
      # get logits and loss for the current context window
      logits, loss = self(idx_cond)
      # focus only on the last time step
      logits = logits[:, -1, :]
      # apply softmax to get probabilities
      softmax = F.softmax(logits, dim=-1)
      # sample from the distribution
      idx_next = torch.multinomial(softmax, num_samples=1)
      # append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1)
    return idx

model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# train eval loop
for steps in range(max_iters):
  
  if steps % eval_interval == 0:
    losses = estimate_loss()
    print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
  # get sample batch from training set
  xb, yb = get_batch('train')

  # evaluate loss and optimize
  logits, loss = m(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()


idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.genereate(idx, max_new_tokens=500)[0].tolist()))