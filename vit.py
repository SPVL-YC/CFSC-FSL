import torch
from torch import nn
from einops import rearrange, repeat

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  
        self.fn = fn  

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)  

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),  
            nn.GELU(),  
            nn.Dropout(dropout),  
            nn.Linear(hidden_dim, dim),  
            nn.Dropout(dropout)  
        )

    def forward(self, x):
        return self.net(x)  


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  
        project_out = not (heads == 1 and dim_head == dim)  

        self.heads = heads  
        self.scale = dim_head ** -0.5  

        self.attend = nn.Softmax(dim=-1)  
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),  
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()  

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)  
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)  
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  
        attn = self.attend(dots)  
        out = torch.matmul(attn, v)  
        out = rearrange(out, 'b h n d -> b n (h d)')  
        return self.to_out(out), attn  
    

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])  
        for _ in range(depth):  
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),  
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))  
            ]))

    def forward(self, x):
        att = None  
        for attn, ff in self.layers:  
            att_x, att = attn(x)  
            x = att_x + x  
            x = ff(x) + x  
        return x, att.mean(dim=1)  

class WordEmbTransformers(nn.Module):
    """Section 2.4.2: token-wise Linear(768, d), BatchNorm, GELU, Dropout."""

    def __init__(self, feature_dim, dropout, input_dim=768):
        super().__init__()
        self.feature_dim = feature_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, attention_mask=None):
        if x.ndim != 3:
            raise ValueError("Text mapping expects [B, L, 768], including CLS")
        if attention_mask is None:
            attention_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        if attention_mask.shape != x.shape[:2]:
            raise ValueError("Text mask must have shape [B, L]")
        valid = attention_mask.to(device=x.device, dtype=torch.bool).reshape(-1)
        if not valid.any():
            raise ValueError("Text mapping needs at least one valid token")
        flat = x.reshape(-1, x.shape[-1])
        mapped = self.fc(flat[valid])
        output = mapped.new_zeros((flat.shape[0], self.feature_dim))
        output[valid] = mapped
        return output.reshape(x.shape[0], x.shape[1], self.feature_dim)


def map_episode_text(encoder, class_bank, class_mask, dataset_bank, dataset_mask, class_ids):
    """Select real class IDs in query order; map the shared dataset text once."""
    device = next(encoder.parameters()).device
    indices = torch.as_tensor(class_ids, dtype=torch.long, device=class_bank.device)
    selected_mask = class_mask.index_select(0, indices).to(device)
    class_tokens = encoder(class_bank.index_select(0, indices).to(device), selected_mask)
    shared_mask = dataset_mask.to(device)
    shared_tokens = encoder(dataset_bank.to(device), shared_mask)
    if shared_tokens.shape[0] != 1:
        raise ValueError("Expected one concatenated dataset-level description")
    batch_size = class_tokens.shape[0]
    return (class_tokens, shared_tokens.expand(batch_size, -1, -1),
            selected_mask, shared_mask.expand(batch_size, -1))


def compose_text_tokens(class_tokens, dataset_tokens, alpha, beta,
                        class_mask=None, dataset_mask=None):
    """Eq. (16)-(17): weighted token concatenation and weighted CLS sum.

    Alpha and beta are the independent attention-derived weights of Eq. (9).
    They are NOT forced to sum to one. CLS tokens remain in the CMF sequence.
    """
    if class_tokens.ndim != 3 or dataset_tokens.ndim != 3:
        raise ValueError("Expected [B, L, D] text token sequences")
    batch_size = class_tokens.shape[0]
    if dataset_tokens.shape[0] != batch_size or dataset_tokens.shape[2] != class_tokens.shape[2]:
        raise ValueError("Class and dataset text must share batch and feature dimensions")
    if alpha.numel() != batch_size or beta.numel() != batch_size:
        raise ValueError("Expected one alpha and beta per sample")
    alpha = alpha.to(class_tokens).reshape(batch_size, 1, 1)
    beta = beta.to(dataset_tokens).reshape(batch_size, 1, 1)
    weighted_class = alpha * class_tokens
    weighted_dataset = beta * dataset_tokens
    tokens = torch.cat((weighted_class, weighted_dataset), dim=1)
    global_text = weighted_class[:, 0, :] + weighted_dataset[:, 0, :]
    masks = []
    for sequence, mask in ((class_tokens, class_mask), (dataset_tokens, dataset_mask)):
        if mask is None:
            mask = torch.ones(sequence.shape[:2], dtype=torch.bool, device=sequence.device)
        if mask.shape != sequence.shape[:2]:
            raise ValueError("Text mask must match its token sequence")
        masks.append(mask.to(device=sequence.device, dtype=torch.bool))
    return tokens, global_text, torch.cat(masks, dim=1)


class CrossModalAttention(nn.Module):
    """CMF, Eq. (20)-(22): visual Q, text K/V, residual + LayerNorm.

    visual_tokens: [B, P+1, d], CLS first; text_tokens: [B, L, d].
    Returns the enhanced CLS [B, d]. Diagnostics optionally include the full
    enhanced sequence and the normalized attention before dropout.
    """

    def __init__(self, feat_dim=100, attn_dim=100, dropout=0.1):
        super().__init__()
        self.attn_dim = attn_dim
        self.W_q = nn.Linear(feat_dim, attn_dim, bias=False)
        self.W_k = nn.Linear(feat_dim, attn_dim, bias=False)
        self.W_v = nn.Linear(feat_dim, attn_dim, bias=False)
        self.W_o = nn.Linear(attn_dim, feat_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feat_dim)
        for projection in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(projection.weight)

    def forward(self, visual_tokens, text_tokens, text_mask=None, return_details=False):
        if visual_tokens.ndim != 3 or text_tokens.ndim != 3:
            raise ValueError("CMF requires token sequences [B, L, D], not global vectors")
        if visual_tokens.shape[0] != text_tokens.shape[0]:
            raise ValueError("Visual and text batches must match")
        Q = self.W_q(visual_tokens)
        K = self.W_k(text_tokens)
        V = self.W_v(text_tokens)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.attn_dim ** 0.5)
        if text_mask is not None:
            if text_mask.shape != text_tokens.shape[:2]:
                raise ValueError("Text mask must have shape [B, L]")
            text_mask = text_mask.to(device=scores.device, dtype=torch.bool)
            if not text_mask.any(dim=-1).all():
                raise ValueError("Every sample needs at least one valid text token")
            scores = scores.masked_fill(~text_mask[:, None, :], float("-inf"))
        weights = scores.softmax(dim=-1) 
        context = torch.matmul(self.dropout(weights), V)
        enhanced = self.norm(visual_tokens + self.W_o(context))
        global_feature = enhanced[:, 0, :]
        if return_details:
            return global_feature, enhanced, weights
        return global_feature
