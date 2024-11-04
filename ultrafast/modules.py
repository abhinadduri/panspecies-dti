import torch
import wandb
from torch import nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
     
class Attention(nn.Module):
    """
    A single layer of multi-head self-attention.
    """
    def __init__(self, embed_dim, num_heads, head_dim, dropout=0.0, **kwargs):
        super().__init__()
        self.heads = num_heads
        self.inner_dim = num_heads * head_dim
        self.to_qkv = nn.Linear(embed_dim, self.inner_dim * 3, bias=False)
        self.multihead_attn = nn.MultiheadAttention(self.inner_dim, num_heads, dropout=dropout, batch_first=True, **kwargs)

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        return self.multihead_attn(q, k , v)[0]

class Transformer(nn.Module):
    """
    Layers of self-attention after appending a CLS token to the input.

    The input to this module is expected to be the frozen embeddings of a protein foundation model,
    such as ProtBert or ESM. 

    `dim` is the dimension of the embeddings.
    `depth` is the number of transformer layers.
    `num_heads` is the number of heads in the multi-head attention.
    `head_dim` is the dimension of the heads.
    `mlp_dim` is the hidden dimension of the feed forward network. 
    """
    def __init__(self, dim, depth, num_heads, head_dim, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, num_heads, head_dim, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            # import pdb; pdb.set_trace()
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class TargetEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout = 0., out_type="cls"):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))
        self.transformer = Transformer(embedding_dim, num_layers, 8, embedding_dim // 8, hidden_dim, dropout = dropout)
        self.out_type = "cls" if out_type == "cls" else "mean"

    def forward(self, x):
        """
        Returns the embedding of the CLS token after passing through the transformer.
        """
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        cls_tokens = self.cls_token.repeat(b, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)
        return x[:,0] if self.out_type == "cls" else x[:,1:].mean(dim=1)

class AverageNonZeroVectors(torch.nn.Module):
    def __init__(self, eps=1e-8):
        super(AverageNonZeroVectors, self).__init__()
        self.eps = eps

    def forward(self, input_batch):
        non_zero_mask = (input_batch.sum(dim=-1) != 0).float()
        num_non_zero = non_zero_mask.sum(dim=1)
        batch_sum = torch.sum(input_batch * non_zero_mask.unsqueeze(-1), dim=1)
        batch_avg = batch_sum / (num_non_zero.unsqueeze(-1) + self.eps)

        if torch.isnan(batch_avg).any() or torch.isinf(batch_avg).any():
            print("NaN or Inf found in averaging operation")
            batch_avg = torch.nan_to_num(batch_avg, nan=0.0, posinf=1.0, neginf=-1.0)

        return batch_avg

# from https://github.com/facebookresearch/deit/blob/main/patchconvnet_models.py
class Learned_Aggregation_Layer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_scale: float = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_avg: bool = False,
        rope: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim: int = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale if qk_scale is not None else head_dim**-0.5

        self.cls_tokenizer = None
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.use_avg = use_avg
        if use_avg == True:
            self.cls_token = None
            self.cls_tokenizer = AverageNonZeroVectors()
        self.pos_embedding = None
        if rope:
            self.pos_embedding = RotaryEmbedding(dim = head_dim//2) # https://github.com/lucidrains/rotary-embedding-torch/issues/4
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.id = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B, N, C = x.shape
        if self.use_avg == False:
            cls_tokens = self.cls_token.repeat(B, 1, 1)
        else:
            cls_tokens = self.cls_tokenizer(x)
        q = self.q(cls_tokens).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.pos_embedding is not None:
            k = self.pos_embedding.rotate_queries_or_keys(k)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)
        attn = self.id(attn)
        attn = attn.softmax(dim=-1)
        soft_attn = attn.clone().reshape(B,self.num_heads, N) 
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls.squeeze(), soft_attn

class LearnedAgg_Projection(nn.Module):
    def __init__(self, target_dim, latent_dim, activation, num_heads=1, attn_drop=0., proj_drop=0., use_avg=False, rope=False):
        super().__init__()
        self.proj = Learned_Aggregation_Layer(target_dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop, use_avg=use_avg, rope=rope)
        self.linear = nn.Linear(target_dim, latent_dim)
        self.non_linearity = activation()

    def forward(self, x):
        proj, attn_head = self.proj(x)
        proj = self.linear(proj)
        return self.non_linearity(proj), attn_head


class LargeProteinProjection(nn.Module):
    def __init__(self, target_dim, latent_dim, activation, num_heads=1, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.proj = Learned_Aggregation_Layer(target_dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)
        self.res = nn.Sequential(
                nn.LayerNorm(target_dim),
                activation(),
                nn.Linear(target_dim, latent_dim),
                nn.Dropout(proj_drop),
                nn.LayerNorm(latent_dim),
                activation(),
                nn.Linear(latent_dim, latent_dim),
                nn.Dropout(proj_dropout),
                nn.LayerNorm(latent_dim),
                activation(),
                nn.Linear(latent_dim, latent_dim),
                nn.Dropout(proj_dropout),
                nn.LayerNorm(latent_dim),
            )
        self.non_linearity = activation()

    def forward(self, x):
        proj, attn_head = self.proj(x)
        proj = self.res(proj)
        return self.non_linearity(proj), attn_head

#from https://github.com/HannesStark/protein-localization/blob/master/models/light_attention.py
class LightAttention(nn.Module):
    def __init__(self, embeddings_dim=1024, output_dim=11, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25):
        super(LightAttention, self).__init__()

        self.feature_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                               padding=kernel_size // 2)

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            nn.Linear(2 * embeddings_dim, embeddings_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(embeddings_dim)
        )

        self.output = nn.Linear(embeddings_dim, output_dim)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        mask = (torch.abs(x).sum(dim=-1) == 0).float()
        x = x.permute(0,2,1)
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        o = self.dropout(o)  # [batch_gsize, embeddings_dim, sequence_length]
        attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]

        # mask out the padding to which we do not want to pay any attention (we have the padding because the sequences have different lenghts).
        # This padding is added by the dataloader when using the padded_permuted_collate function in utils/general.py
        attention = attention.masked_fill(mask[:, None, :] == False, -1e9)

        # code used for extracting embeddings for UMAP visualizations
        # extraction =  torch.sum(x * self.softmax(attention), dim=-1)
        # extraction = self.id0(extraction)

        o1 = torch.sum(o * self.softmax(attention), dim=-1)  # [batchsize, embeddings_dim]
        o2, _ = torch.max(o, dim=-1)  # [batchsize, embeddings_dim]
        o = torch.cat([o1, o2], dim=-1)  # [batchsize, 2*embeddings_dim]
        o = self.linear(o)  # [batchsize, 32]
        return self.output(o), attention  # [batchsize, output_dim]

class LightAttention_Projection(nn.Module):
    def __init__(self, target_dim, latent_dim, activation, dropout=0., conv_dropout=0.):
        super().__init__()
        self.proj = LightAttention(embeddings_dim=target_dim, output_dim=target_dim, dropout=dropout, conv_dropout=conv_dropout)
        self.linear = nn.Linear(target_dim, latent_dim)
        self.non_linearity = activation()

    def forward(self, x):
        proj, attn_head = self.proj(x)
        proj = self.linear(proj)
        return self.non_linearity(proj), attn_head

class HugeProteinProjection(nn.Module):
    def __init__(self, target_dim, latent_dim, activation, num_heads=1, attn_drop=0., proj_drop=0.):
        super().__init__()
        raise NotImplementedError("This module is not implemented yet. Still need to move from model.py")

    def forward(self, x):
        raise NotImplementedError("This module is not implemented yet. Still need to move from model.py")
