import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class MultiHeadAttention(nn.Module):
  def __init__(self):
    super().__init__()
    self.key = nn.Linear(n_embd, n_embd, bias=False)
    self.query = nn.Linear(n_embd, n_embd, bias=False)
    self.value = nn.Linear(n_embd, n_embd, bias=False)
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)
    
    self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    if not self.flash:
      print("Using custom attention")
      self.register_buffer('bias', torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

  def forward(self, x):
    B, T, C = x.size()

    # calculate key, query, value by splitting the embedding into separateheads 
    k = self.key(x).view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, n_head, T, head_size)
    q = self.query(x).view(B, T, n_head, C // n_head).transpose(1, 2) # (B, n_head, T, head_size)
    v = self.value(x).view(B, T, n_head, C // n_head).transpose(1, 2) # (B, n_head, T, head_size)
    
    if self.flash:
      out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.2 if self.training else 0, is_causal=True)
    else:
      wei = q @ k.transpose(-2, -1) * C**-0.5
      wei = wei.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
      wei = F.softmax(wei, dim=-1)
      wei = self.dropout(wei)
      out = wei @ v

      # rearrange the output back to the original shape
    out = out.transpose(1, 2).contiguous().view(B, T, C)
    
    return self.dropout(self.proj(out))

class FeedForward(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd),
      nn.GELU(),
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
  def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
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
