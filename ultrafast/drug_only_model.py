import torch
import wandb
from torch import nn
import torch.nn.functional as F
from ultrafast.contrastive_loss import MarginScheduledLossFunction, InfoNCELoss

import pytorch_lightning as pl
import torchmetrics

class DrugOnlyLightning(pl.LightningModule):
    def __init__(
        self,
        drug_dim=2048,
        latent_dim=1024,
        activation=nn.LeakyReLU,
        classify=True,
        dropout=0.05,
        lr=1e-4,
        contrastive=False,
        InfoNCEWeight=0,
        args=None,
    ):
        super().__init__()

        self.automatic_optimization = False # We will handle the optimization step ourselves
        self.drug_dim = drug_dim
        self.latent_dim = latent_dim
        self.activation = activation

        self.classify = classify
        assert contrastive == False, "Contrastive learning not implemented for Ligand only"
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


        if 'model_size' in args and args.model_size == "huge":  # override the above settings and use a large model for drug and target
            self.drug_projector = nn.ModuleDict({
                'proj': nn.Linear(self.drug_dim, 1260),
                'res': nn.ModuleList([nn.Sequential(nn.Linear(1260, 1260), self.activation(), nn.Dropout(dropout), nn.LayerNorm(1260)) for _ in range(6)]),
                'out': nn.Linear(1260, self.latent_dim),
            })

        if 'model_size' in args and args.model_size == "mega":  # override the above settings and use a large model for drug and target
            self.drug_projector = nn.ModuleDict({
                'proj': nn.Linear(self.drug_dim, 2048),
                'res': nn.ModuleList([nn.Sequential(nn.Linear(2048, 2048), self.activation(), nn.Dropout(dropout), nn.LayerNorm(2048)) for _ in range(8)]),
                'out': nn.Linear(2048, self.latent_dim),
            })

        self.final_layer = nn.Linear(self.latent_dim, 1)

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

    def forward(self, drug):
        model_size = self.args.model_size
        sigmoid_scalar = self.args.sigmoid_scalar
        if model_size == 'huge' or model_size == 'mega':
            y = self.drug_projector['proj'](drug)
            for layer in self.drug_projector['res']:
                y = y + layer(y)
            drug_projection = self.drug_projector['out'](y)
        else:
            drug_projection = self.drug_projector(drug)

        output = self.final_layer(drug_projection)

        if self.classify:
            similarity = sigmoid_scalar * output
        else:
            similarity = output

        return drug_projection, similarity

    def configure_optimizers(self):
        optimizers = []
        lr_schedulers = []
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0= self.args.lr_t0
            )
        optimizers.append(optimizer)
        lr_schedulers.append(lr_scheduler)
        return (optimizers, lr_schedulers)

    def non_contrastive_step(self, batch, train=True):
        drug, protein, label = batch
        drug, similarity = self.forward(drug)


        if self.classify:
            similarity = torch.squeeze(self.sigmoid(similarity))

        loss = self.loss_fct(similarity, label) 
        infoloss = 0
        if self.InfoNCEWeight > 0:
            infoloss = self.InfoNCEWeight * self.infoNCE_loss_fct(drug, drug, label)

        if train:
            return loss * self.CEWeight, infoloss
        else:
            return loss, infoloss, similarity

    def training_step(self, batch, batch_idx):
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
            if model_size == 'huge' or model_size == 'mega':
                y = self.drug_projector['proj'](x)
                for layer in self.drug_projector['res']:
                    y = y + layer(y)
                drug_projection = self.drug_projector['out'](y)
            else:
                drug_projection = self.drug_projector(x)
            return drug_projection

        elif sample_type == "target":
            raise NotImplementedError("Target embedding not implemented for this model")
