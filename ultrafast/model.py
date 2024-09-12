import torch
import wandb
from torch import nn
import torch.nn.functional as F
from ultrafast.contrastive_loss import MarginScheduledLossFunction, InfoNCELoss

import pytorch_lightning as pl
import torchmetrics

class FocalLoss(nn.Module):
    ### https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
    def __init__(self,
                 reduction="mean",
                 alpha=-1,
                 gamma=2):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

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
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim: int = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale if qk_scale is not None else head_dim**-0.5

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
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
        cls_tokens = self.cls_token.repeat(B, 1, 1)
        q = self.q(cls_tokens).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)
        attn = self.id(attn)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls.squeeze()

class AverageNonZeroVectors(torch.nn.Module):
    def __init__(self):
        super(AverageNonZeroVectors, self).__init__()

    def forward(self, input_batch):
        # Calculate the mask for non-zero vectors
        non_zero_mask = (input_batch.sum(dim=-1) != 0).float()

        # Calculate the total number of non-zero vectors in the batch
        num_non_zero = non_zero_mask.sum(dim=1)

        # Calculate the sum of non-zero vectors in the batch
        batch_sum = torch.sum(input_batch * non_zero_mask.unsqueeze(-1), dim=1)

        # Calculate the average of non-zero vectors in the batch
        batch_avg = batch_sum / num_non_zero.unsqueeze(-1)

        return batch_avg

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
        contrastive=False,
        InfoNCEWeight=0,
        args=None,
    ):
        super().__init__()

        self.automatic_optimization = False # We will handle the optimization step ourselves
        self.drug_dim = drug_dim
        self.target_dim = target_dim
        self.latent_dim = latent_dim
        self.activation = activation

        self.classify = classify
        self.contrastive = contrastive
        self.args = args

        if args.drug_layers == 1:
            self.drug_projector = nn.Sequential(
                nn.Linear(self.drug_dim, self.latent_dim), self.activation()
            )
            nn.init.xavier_normal_(self.drug_projector[0].weight)
        elif args.drug_layers == 2:
            self.drug_projector = nn.Sequential(
                nn.Linear(self.drug_dim, 1260), self.activation(),
                nn.Linear(1260, self.latent_dim), self.activation()
            )
            nn.init.xavier_normal_(self.drug_projector[0].weight)
            nn.init.xavier_normal_(self.drug_projector[2].weight)


        
        if 'prot_proj' not in args or args.prot_proj == "avg":
            protein_projector=nn.Sequential(AverageNonZeroVectors(), nn.Linear(self.target_dim, self.latent_dim))
        elif args.prot_proj == "transformer":
            protein_projector = TargetEmbedding( self.target_dim, self.latent_dim, num_layers_target, dropout=dropout, out_type=args.out_type)
        elif args.prot_proj == "agg":
            protein_projector = nn.Sequential(Learned_Aggregation_Layer(self.target_dim, num_heads=self.args.num_heads_agg, attn_drop=dropout, proj_drop=dropout), nn.Linear(self.target_dim, self.latent_dim))

        self.target_projector = nn.Sequential(
                protein_projector,
                self.activation()
        )
        if 'prot_proj' not in args or args.prot_proj == "avg":
            nn.init.xavier_normal_(self.target_projector[0][1].weight)

        if self.classify:
            self.sigmoid = nn.Sigmoid()
            self.val_accuracy = torchmetrics.Accuracy(task='binary')
            self.val_aupr = torchmetrics.AveragePrecision(task='binary')
            self.val_auroc = torchmetrics.AUROC(task='binary')
            self.val_f1 = torchmetrics.F1Score(task='binary')
            self.metrics = {
                "acc": self.val_accuracy,
                "aupr": self.val_aupr,
                "auroc": self.val_auroc,
                "f1": self.val_f1,
            }
            self.loss_fct = torch.nn.BCELoss()
            if self.args.loss_type == "focal":
                self.loss_fct = FocalLoss()
        else:
            self.val_mse = torchmetrics.MeanSquaredError()
            self.val_pcc = torchmetrics.PearsonCorrCoef()
            self.metrics = {"mse": self.val_mse, "pcc": self.val_pcc}
            self.loss_fct = torch.nn.MSELoss()

        if self.contrastive:
            self.contrastive_loss_fct = MarginScheduledLossFunction(
                    M_0 = args.margin_max,
                    N_epoch = args.epochs,
                    N_restart = args.margin_t0,
                    update_fn = args.margin_fn
                    )
        # instantiate InfoNCE loss function as 0 for now
        self.InfoNCEWeight = InfoNCEWeight
        if self.InfoNCEWeight:
            self.infoNCE_loss_fct = InfoNCELoss(temperature=args.InfoNCETemp if 'InfoNCETemp' in args else 0.5) 

        self.CEWeight = 1 if 'CEWeight' not in args else args.CEWeight
                    

        self.save_hyperparameters()

        self.val_step_outputs = []
        self.val_step_targets = []
        self.test_step_outputs = []
        self.test_step_targets = []

    def forward(self, drug, target):
        drug_projection = self.drug_projector(drug)
        # Add a batch dimension if it's missing
        if target.dim() == 2:
            target = target.unsqueeze(0)
        target_projection = self.target_projector(target)

        if self.classify:
            similarity = F.cosine_similarity(
                drug_projection, target_projection
            )
        else:
            similarity = torch.bmm(
                drug_projection.view(-1, 1, self.latent_dim),
                target_projection.view(-1, self.latent_dim, 1),
            ).squeeze()

        return drug_projection, target_projection, similarity

    def configure_optimizers(self):
        optimizers = []
        lr_schedulers = []
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0= self.args.lr_t0
            )
        optimizers.append(optimizer)
        lr_schedulers.append(lr_scheduler)
        if self.contrastive:
            opt_contrastive = torch.optim.Adam(self.parameters(), lr=self.args.clr)
            lr_scheduler_contrastive = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt_contrastive, T_0=self.args.clr_t0
            )
            optimizers.append(opt_contrastive)
            lr_schedulers.append(lr_scheduler_contrastive)
        return (optimizers, lr_schedulers)

    def contrastive_step(self, batch):
        anchor, positive, negative = batch

        anchor_projection = self.target_projector(anchor)
        positive_projection = self.drug_projector(positive)
        negative_projection = self.drug_projector(negative)

        
        loss = self.contrastive_loss_fct(anchor_projection, positive_projection, negative_projection)
        return loss

    def non_contrastive_step(self, batch, train=True):
        drug, protein, label = batch
        drug, protein, similarity = self.forward(drug, protein)


        if self.classify:
            similarity = torch.squeeze(self.sigmoid(similarity))

        loss = self.loss_fct(similarity, label) 
        infoloss = 0
        if self.InfoNCEWeight > 0:
            infoloss = self.InfoNCEWeight * self.infoNCE_loss_fct(drug, protein, label)

        if train:
            return loss * self.CEWeight, infoloss
        else:
            return loss, infoloss, similarity

    def training_step(self, batch, batch_idx):
        if self.contrastive and self.current_epoch % 2 == 1:
            _, con_opt = self.optimizers()
            con_opt.zero_grad()
            loss = self.contrastive_step(batch)
            self.manual_backward(loss)
            con_opt.step()
            self.log("train/contrastive_loss", loss, sync_dist=True if self.trainer.num_devices > 1 else False)
        else:
            if self.contrastive:
                opt, _ = self.optimizers()
            else:
                opt = self.optimizers()
            opt.zero_grad()
            loss,infoloss = self.non_contrastive_step(batch)
            self.manual_backward(loss+infoloss)
            opt.step()
            self.log("train/loss", loss, sync_dist=True if self.trainer.num_devices > 1 else False)
            if self.InfoNCEWeight > 0:
                self.log("train/info_loss", infoloss, sync_dist=True if self.trainer.num_devices > 1 else False)

        return loss

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        if self.contrastive:
            if self.current_epoch % 2 == 0: # supervised learning epoch
                sch[0].step()
                self.log("train/lr", sch[0].get_lr()[0], sync_dist=True if self.trainer.num_devices > 1 else False)
            else: # contrastive learning epoch
                sch[1].step()
                self.contrastive_loss_fct.step()
                self.log("train/triplet_margin", self.contrastive_loss_fct.margin, sync_dist=True if self.trainer.num_devices > 1 else False)
                self.log("train/contrastive_lr", sch[1].get_lr()[0], sync_dist=True if self.trainer.num_devices > 1 else False)
        else:
            self.log("train/lr", sch.get_lr()[0], sync_dist=True if self.trainer.num_devices > 1 else False)
            sch.step()

    def validation_step(self, batch, batch_idx):
        if self.global_step == 0 and self.global_rank == 0 and not self.args.no_wandb:
            wandb.define_metric("val/aupr", summary="max")
        _, _, label = batch
        loss, infoloss, similarity = self.non_contrastive_step(batch, train=False)
        self.log("val/loss", loss, sync_dist=True if self.trainer.num_devices > 1 else False)
        if self.InfoNCEWeight > 0:
            self.log("val/info_loss", infoloss, sync_dist=True if self.trainer.num_devices > 1 else False)

        self.val_step_outputs.extend(similarity)
        self.val_step_targets.extend(label)

        return {"loss": loss, "preds": similarity, "target": label}

    def on_validation_epoch_end(self):
        for name, metric in self.metrics.items():
            if self.classify:
                metric(torch.Tensor(self.val_step_outputs), torch.Tensor(self.val_step_targets).to(torch.int))
            else:
                metric(torch.Tensor(self.val_step_outputs).cuda(), torch.Tensor(self.val_step_targets).to(torch.float).cuda())
            self.log(f"val/{name}", metric, on_step=False, on_epoch=True, sync_dist=True if self.trainer.num_devices > 1 else False)

        self.val_step_outputs.clear()
        self.val_step_targets.clear()

    def test_step(self, batch, batch_idx):
        _, _, label = batch
        _, _, similarity = self.non_contrastive_step(batch, train=False)

        self.test_step_outputs.extend(similarity)
        self.test_step_targets.extend(label)

        return {"preds": similarity, "target": label}

    def on_test_epoch_end(self):
        for name, metric in self.metrics.items():
            if self.classify:
                metric(torch.Tensor(self.test_step_outputs), torch.Tensor(self.test_step_targets).to(torch.int))
            else:
                metric(torch.Tensor(self.test_step_outputs).cuda(), torch.Tensor(self.test_step_targets).to(torch.float).cuda())
            self.log(f"test/{name}", metric, on_step=False, on_epoch=True, sync_dist=True if self.trainer.num_devices > 1 else False)

        self.test_step_outputs.clear()
        self.test_step_targets.clear()

    def embed(self, x, sample_type="drug"):
        if sample_type == "drug":
            return self.drug_projector(x)
        elif sample_type == "target":
            return self.target_projector(x)


def main():
    from featurizers import ProtBertFeaturizer
    protbert = ProtBertFeaturizer()

    out = torch.stack([protbert("AGGA"), protbert("AGGA")], dim=0)
    model = TargetEmbedding(1024, 1024, 2)

    # Testing to make sure a forward pass works
    print(model(out).shape)

if __name__ == "__main__":
    main()



