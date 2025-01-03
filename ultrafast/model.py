import torch
import wandb
from torch import nn
import torch.nn.functional as F
from ultrafast.contrastive_loss import MarginScheduledLossFunction, InfoNCELoss
from ultrafast.modules import AverageNonZeroVectors, Learned_Aggregation_Layer, TargetEmbedding, LargeDrugProjector, LargeProteinProjector

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

class DrugTargetCoembeddingLightning(pl.LightningModule):
    def __init__(
        self,
        drug_dim=2048,
        target_dim=100,
        latent_dim=1024,
        activation=nn.LeakyReLU,
        classify=True,
        num_layers_target=1,
        dropout=0.05,
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

        if 'model_size' in args and args.model_size == "large":  # override the above settings and use a large model for drug and target
            self.drug_projector = LargeDrugProjector(self.drug_dim, self.latent_dim, self.activation, dropout)
            protein_projector = LargeProteinProjector(self.target_dim, self.latent_dim, self.activation, self.args.num_heads_agg, attn_drop=dropout, proj_drop=dropout)

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
            self.val_krc = torchmetrics.KendallRankCorrCoef()
            self.metrics = {"mse": self.val_mse, "pcc": self.val_pcc, "kendalls_tau": self.val_krc}
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
        model_size = self.args.model_size
        sigmoid_scalar = self.args.sigmoid_scalar

        drug_projection = self.drug_projector(drug)

        # Add a batch dimension if it's missing
        if target.dim() == 2:
            target = target.unsqueeze(0)

        target_projection = self.target_projector(target)

        if self.classify:
            similarity = sigmoid_scalar * F.cosine_similarity(
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=0.0 if not hasattr(self.args, 'weight_decay') else self.args.weight_decay) 
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
        model_size = self.args.model_size
        if sample_type == "drug":
            drug_projection = self.drug_projector(x)
            return drug_projection
        elif sample_type == "target":
            target_projection = self.target_projector(x)
            return target_projection


def main():
    from featurizers import ProtBertFeaturizer
    protbert = ProtBertFeaturizer()

    out = torch.stack([protbert("AGGA"), protbert("AGGA")], dim=0)
    model = TargetEmbedding(1024, 1024, 2)

    # Testing to make sure a forward pass works
    print(model(out).shape)

if __name__ == "__main__":
    main()



