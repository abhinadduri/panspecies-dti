import torch
from torch import nn

import pytorch_lightning as pl
import torchmetrics

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
    def __init__(self, embed_dim, num_heads, dim_heads, dropout=0.0, **kwargs):
        super().__init__()
        self.heads = num_heads
        self.inner_dim = num_heads * dim_head
        self.to_qkv = nn.Linear(embed_dim, inner_dim * 3, bias=False)
        self.multihead_attn = nn.MultiheadAttention(inner_dim, num_heads, dropout=dropout, **kwargs)

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        return self.multihead_attn(q, k , v)[0]

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, num_heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class TargetEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout = 0.):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.input_size))
        self.transformer = Transformer(embedding_dim, num_layers, 8, 64, hidden_dim, dropout = dropout)

    def forward(self, x):
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)
        return x[:,0]

class DrugTargetCoembeddingLightning(pl.LightningModule):
    def __init__(
        self,
        drug_dim=2048,
        target_dim=100,
        latent_dim=1024,
        activation=nn.ReLU,
        classify=True,
        num_layers_target=1,
        dropout=0,
        lr=1e-4,
    ):
        super().__init__()

        self.drug_dim = drug_dim
        self.target_dim = drug_dim
        self.latent_dim = latent_dim
        self.activation = activation

        self.classify = classify
        self.lr = lr

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_dim, self.latent_dim), self.activation()
        )

        self.target_projector = nn.Sequential(
            TargetEmbedding( self.target_dim, self.latent_dim, num_layers_target, dropout=dropout), self.activation()
        )

        if self.classify:
            self.val_accuracy = torchmetrics.Accuracy()
            self.val_aupr = torchmetrics.AveragePrecision()
            self.val_auroc = torchmetrics.AUROC()
            self.val_f1 = torchmetrics.F1Score()
            self.metrics = {
                "acc": self.val_accuracy,
                "aupr": self.val_aupr,
                "auroc": self.val_auroc,
                "f1": self.val_f1,
            }
        else:
            self.val_mse = torchmetrics.MeanSquaredError()
            self.val_pcc = torchmetrics.PearsonCorrCoef()
            self.metrics = {"mse": self.val_mse, "pcc": self.val_pcc}

    def forward(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        if self.classify:
            similarity = nn.CosineSimilarity()(
                drug_projection, target_projection
            )
        else:
            similarity = torch.bmm(
                drug_projection.view(-1, 1, self.latent_dim),
                target_projection.view(-1, self.latent_dim, 1),
            ).squeeze()

        return similarity

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        drug, protein, label = train_batch
        similarity = self.forward(drug, protein)

        if self.classify:
            sigmoid = torch.nn.Sigmoid()
            similarity = torch.squeeze(sigmoid(similarity))
            loss_fct = torch.nn.BCELoss()
        else:
            loss_fct = torch.nn.MSELoss()

        loss = loss_fct(similarity, label)
        self.log("train/loss", loss)

        return loss

    def validation_step(self, train_batch, batch_idx):
        drug, protein, label = train_batch
        similarity = self.forward(drug, protein)

        if self.classify:
            sigmoid = torch.nn.Sigmoid()
            similarity = torch.squeeze(sigmoid(similarity))
            loss_fct = torch.nn.BCELoss()
        else:
            loss_fct = torch.nn.MSELoss()

        loss = loss_fct(similarity, label)
        self.log("val/loss", loss)
        return {"loss": loss, "preds": similarity, "target": label}

    def validation_step_end(self, outputs):
        for name, metric in self.metrics.items():
            metric(outputs["preds"], outputs["target"])
            self.log(f"val/{name}", metric)
