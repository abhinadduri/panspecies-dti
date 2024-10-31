import torch
import wandb
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ultrafast.modules import LearnedAgg_Projection, LargeProteinProjection, TargetEmbedding, Attention, Learned_Aggregation_Layer, AverageNonZeroVectors
from ultrafast.loss import MarginScheduledLossFunction, InfoNCELoss, AttentionGuidanceLoss, PatternDecorrelationLoss

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
            self.target_projector=nn.Sequential(AverageNonZeroVectors(), nn.Linear(self.target_dim, self.latent_dim), self.activation())
        elif args.prot_proj == "transformer":
            protein_projector = TargetEmbedding( self.target_dim, self.latent_dim, num_layers_target, dropout=dropout, out_type=args.out_type)
        elif args.prot_proj == "agg":
            self.target_projector = LearnedAgg_Projection(self.target_dim, self.latent_dim, self.activation, num_heads=args.num_heads_agg, attn_drop=dropout, proj_drop=dropout, use_avg=args.agg_use_avg)

        if 'model_size' in args and args.model_size == "large":  # override the above settings and use a large model for drug and target
            self.drug_projector = nn.Sequential(
                nn.Linear(self.drug_dim, 1260),
                self.activation(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(1260),
                nn.Linear(1260, self.latent_dim),
                self.activation(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(self.latent_dim),
                nn.Linear(self.latent_dim, self.latent_dim),
                self.activation(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(self.latent_dim),
                nn.Linear(self.latent_dim, self.latent_dim),
                self.activation()
            )
            nn.init.xavier_normal_(self.drug_projector[0].weight)
            nn.init.xavier_normal_(self.drug_projector[4].weight)
            nn.init.xavier_normal_(self.drug_projector[-2].weight)
            nn.init.xavier_normal_(self.drug_projector[-6].weight)

            self.target_projector = LargeProteinProjection(self.target_dim, self.latent_dim, self.activation, num_heads=args.num_heads_agg, attn_drop=dropout, proj_drop=dropout)

        if 'model_size' in args and args.model_size == "huge":  # override the above settings and use a large model for drug and target
            self.drug_projector = nn.ModuleDict({
                'proj': nn.Linear(self.drug_dim, 1260),
                'res': nn.ModuleList([nn.Sequential(nn.Linear(1260, 1260), self.activation(), nn.Dropout(dropout), nn.LayerNorm(1260)) for _ in range(6)]),
                'out': nn.Linear(1260, self.latent_dim),
            })

            self.target_projector = nn.ModuleDict({
                'attn': nn.Sequential(
                    Attention(self.target_dim, self.args.num_heads_agg, self.target_dim // self.args.num_heads_agg, dropout=dropout),
                    Learned_Aggregation_Layer(self.target_dim, attn_drop=dropout, proj_drop=dropout, num_heads=self.args.num_heads_agg),
                ),
                'proj': nn.Linear(self.target_dim, 1260),
                'res': nn.ModuleList([nn.Sequential(nn.Linear(1260, 1260), self.activation(), nn.Dropout(dropout), nn.LayerNorm(1260)) for _ in range(4)]),
                'out': nn.Linear(1260, self.latent_dim),
            })

        if 'model_size' in args and args.model_size == "mega":  # override the above settings and use a large model for drug and target
            self.drug_projector = nn.ModuleDict({
                'proj': nn.Linear(self.drug_dim, 2048),
                'res': nn.ModuleList([nn.Sequential(nn.Linear(2048, 2048), self.activation(), nn.Dropout(dropout), nn.LayerNorm(2048)) for _ in range(8)]),
                'out': nn.Linear(2048, self.latent_dim),
            })

            self.target_projector = nn.ModuleDict({
                'attn': nn.Sequential(
                    Attention(self.target_dim, self.args.num_heads_agg, self.target_dim // self.args.num_heads_agg, dropout=dropout),
                    Attention(self.target_dim, self.args.num_heads_agg, self.target_dim // self.args.num_heads_agg, dropout=dropout),
                    Learned_Aggregation_Layer(self.target_dim, attn_drop=dropout, proj_drop=dropout, num_heads=self.args.num_heads_agg),
                ),
                'proj': nn.Linear(self.target_dim, 2048),
                'res': nn.ModuleList([nn.Sequential(nn.Linear(2048, 2048), self.activation(), nn.Dropout(dropout), nn.LayerNorm(2048)) for _ in range(6)]),
                'out': nn.Linear(2048, self.latent_dim),
            })
        
        if 'prot_proj' not in args or args.prot_proj == "avg":
            nn.init.xavier_normal_(self.target_projector[1].weight)

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

        self.AG = 0
        if args.AG:
            self.AG = args.AG
            self.AG_loss = AttentionGuidanceLoss(loss=args.AG_type)
            self.bs_auc = torchmetrics.AUROC(task="binary")
            self.bs_aupr = torchmetrics.AveragePrecision(task="binary")
            self.bs_iou = torchmetrics.JaccardIndex(task="binary").to(self.device)
            self.metrics['auc_bs'] = self.bs_auc
            self.metrics['aupr_bs'] = self.bs_aupr
            self.metrics['iou_bs'] = self.bs_iou

        self.PDG = 0
        if args.PDG:
            self.PDG = args.PDG
            self.PDG_loss = PatternDecorrelationLoss()

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

        self.log_vals = {'val': {'outputs':[], 'targets':[], 'bs_outputs':None, 'bs_targets': None},
                    'test':{'outputs':[],'targets':[], 'bs_outputs':None, 'bs_targets':None}}

        if 'auc_bs' in self.metrics:
            self.log_vals['val']['bs_outputs'] = []
            self.log_vals['val']['bs_targets'] = []
            self.log_vals['test']['bs_outputs'] = []
            self.log_vals['test']['bs_targets'] = []


    def forward(self, drug, target):
        model_size = self.args.model_size
        if model_size == 'huge' or model_size == 'mega':
            y = self.drug_projector['proj'](drug)
            for layer in self.drug_projector['res']:
                y = y + layer(y)
            drug_projection = self.drug_projector['out'](y)
        else:
            drug_projection = self.drug_projector(drug)

        # Add a batch dimension if it's missing
        if target.dim() == 2:
            target = target.unsqueeze(0)

        attn_head = None
        if model_size == 'huge' or model_size == 'mega':
            z = self.target_projector['attn'](target)
            z = self.target_projector['proj'](z)
            for layer in self.target_projector['res']:
                z = z + layer(z)
            target_projection = self.target_projector['out'](z)
        else:
            if self.args.prot_proj == "agg":
                target_projection, attn_head = self.target_projector(target)
            else:
                target_projection = self.target_projector(target)

        if self.classify:
            similarity = 5 * F.cosine_similarity(
                drug_projection, target_projection
            )
        else:
            similarity = torch.bmm(
                drug_projection.view(-1, 1, self.latent_dim),
                target_projection.view(-1, self.latent_dim, 1),
            ).squeeze()

        return similarity, drug_projection, target_projection, attn_head

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
        if self.args.task != 'binding_site':
            drug, protein, label = batch
        else:
            drug, protein, binding_site, label = batch
        similarity, drug_proj, protein_proj, attn_head = self.forward(drug, protein)

        if self.classify:
            similarity = torch.squeeze(self.sigmoid(similarity * 5))

        loss = self.loss_fct(similarity, label) 
        infoloss = 0
        if self.InfoNCEWeight > 0:
            infoloss = self.infoNCE_loss_fct(drug_proj, protein_proj, label)

        ag_loss = 0
        if self.AG != 0 and self.args.task == 'binding_site':
            ag_loss = self.AG_loss(attn_head, binding_site)

        pdg_loss = 0
        if self.PDG != 0:
            pdg_loss = self.PDG_loss(attn_head)


        if train:
            return loss * self.CEWeight, {'AG':ag_loss, 'PDG':pdg_loss, "InfoNCE": infoloss}
        else:
            return loss, {'AG':ag_loss, 'PDG':pdg_loss, "InfoNCE": infoloss}, similarity, attn_head

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
            loss, oth_losses = self.non_contrastive_step(batch)
            tot_loss = loss + self.PDG * oth_losses['PDG'] + self.AG * oth_losses['AG'] + self.InfoNCEWeight * oth_losses['InfoNCE']
            self.manual_backward(tot_loss)
            opt.step()
            self.log("train/loss", loss, sync_dist=True if self.trainer.num_devices > 1 else False)
            if self.InfoNCEWeight > 0:
                self.log("train/info_loss", oth_losses["InfoNCE"], sync_dist=True if self.trainer.num_devices > 1 else False)
            if self.AG != 0:
                self.log("train/AG_loss", oth_losses['AG'])
            if self.PDG != 0:
                self.log("train/PDG_loss", oth_losses['PDG'])

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
        label = batch[-1]
        loss, oth_losses, similarity, attn_head = self.non_contrastive_step(batch, train=False)
        self.log("val/loss", loss, sync_dist=True if self.trainer.num_devices > 1 else False)
        if self.InfoNCEWeight > 0:
            self.log("val/info_loss", oth_losses["InfoNCE"], sync_dist=True if self.trainer.num_devices > 1 else False)

        if self.AG != 0:
            self.log("val/AG_loss", oth_losses["AG"])

        if self.PDG != 0:
            self.log("val/PDG_loss", oth_losses["PDG"])

        self.log_vals['val']['outputs'].extend(similarity)
        self.log_vals['val']['targets'].extend(label)
        if self.AG != 0 and self.args.task == "binding_site":
            self.log_vals['val']['bs_targets'].extend(batch[-2])
            self.log_vals['val']['bs_outputs'].extend(attn_head[:,0,:])

        return {"loss": loss, "preds": similarity, "target": label}

    def on_validation_epoch_end(self):
        self.compute_metrics("val")

    def test_step(self, batch, batch_idx):
        label = batch[-1]
        _, _, similarity, attn_head = self.non_contrastive_step(batch, train=False)

        self.log_vals['test']['outputs'].extend(similarity)
        self.log_vals['test']['targets'].extend(label)
        if self.AG != 0 and self.args.task == "binding_site":
            self.log_vals['test']['bs_targets'].extend(batch[-2])
            self.log_vals['test']['bs_outputs'].extend(attn_head[:,0,:])

        return {"preds": similarity, "target": label}

    def on_test_epoch_end(self):
        self.compute_metrics("test")

    def compute_metrics(self, stage):
        for name, metric in self.metrics.items():
            if 'bs' in name:
                metric(pad_sequence(self.log_vals[stage]['bs_outputs']).to(self.device), pad_sequence(self.log_vals[stage]['bs_targets']).to(torch.int).to(self.device))
            if self.classify:
                metric(torch.Tensor(self.log_vals[stage]['outputs']).to(self.device), torch.Tensor(self.log_vals[stage]['targets']).to(torch.int).to(self.device))
            else:
                metric(torch.Tensor(self.log_vals[stage]['outputs']).cuda(), torch.Tensor(self.log_vals[stage]['targets']).to(torch.float).cuda())
            self.log(f"{stage}/{name}", metric, on_step=False, on_epoch=True, sync_dist=True if self.trainer.num_devices > 1 else False)
            # print(f"{stage}/{name}:{metric.compute()}")

        self.log_vals[stage]['outputs'].clear()
        self.log_vals[stage]['targets'].clear()
        if isinstance(self.log_vals[stage]['bs_outputs'],list):
            self.log_vals[stage]['bs_outputs'].clear()
            self.log_vals[stage]['bs_targets'].clear()

    def embed(self, x, sample_type="drug"):
        model_size = self.args.model_size
        if sample_type == "drug":
            if model_size == 'huge' or model_size == 'mega':
                y = self.drug_projector['proj'](x)
                for layer in self.drug_projector['res']:
                    y = y + layer(y)
                drug_projection = self.drug_projector['out'](y)
            else:
                drug_projection = self.drug_projector(x)
            return drug_projection

        elif sample_type == "target":
            if model_size == 'huge' or model_size == 'mega':
                z = self.target_projector['attn'](x)
                z = self.target_projector['proj'](z)
                for layer in self.target_projector['res']:
                    z = z + layer(z)
                target_projection = self.target_projector['out'](z)
            else:
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



