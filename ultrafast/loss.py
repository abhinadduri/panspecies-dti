import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial


def sigmoid_cosine_distance_p(x, y, p=1):
    return (1 - F.sigmoid(F.cosine_similarity(x, y))) ** p
    sig = ()

def tanh_decay(M_0, N_epoch, x):
    return M_0 * (1 - np.tanh(2 * x / N_epoch))


def cosine_anneal(M_0, N_epoch, x):
    return 0.5 * M_0 * (1 + np.cos(x * np.pi / N_epoch))


def no_decay(M_0, N_epoch, x):
    return M_0


MARGIN_FN_DICT = {
    "tanh_decay": tanh_decay,
    "cosine_anneal": cosine_anneal,
    "no_decay": no_decay,
}


class MarginScheduledLossFunction:
    def __init__(
        self,
        M_0: float = 0.25,
        N_epoch: float = 50,
        N_restart: float = -1,
        update_fn="tanh_decay",
    ):
        self.M_0 = M_0
        self.N_epoch = N_epoch
        if N_restart == -1:
            self.N_restart = N_epoch
        else:
            self.N_restart = N_restart

        self._step = 0
        self.M_curr = self.M_0

        self._update_fn_str = update_fn
        self._update_margin_fn = self._get_update_fn(update_fn)

        self._update_loss_fn()

    @property
    def margin(self):
        return self.M_curr

    def _get_update_fn(self, fn_string):
        return partial(MARGIN_FN_DICT[fn_string], self.M_0, self.N_restart)

    def _update_loss_fn(self):
        self._loss_fn = torch.nn.TripletMarginWithDistanceLoss(
            distance_function=sigmoid_cosine_distance_p,
            margin=self.M_curr,
        )

    def step(self):
        self._step += 1
        if self._step == self.N_restart:
            self.reset()
        else:
            self.M_curr = self._update_margin_fn(self._step)
            self._update_loss_fn()

    def reset(self):
        self._step = 0
        self.M_curr = self._update_margin_fn(self._step)
        self._update_loss_fn()

    def __call__(self, anchor, positive, negative):
        # logg.debug(anchor, anchor.shape)
        # logg.debug(positive, positive.shape)
        # logg.debug(negative, negative.shape)
        return self._loss_fn(anchor, positive, negative)

class AttentionGuidanceLoss(torch.nn.Module):
    def __init__(self, loss='mse_loss'):
        super(AttentionGuidanceLoss, self).__init__()

        self.loss = F.mse_loss if loss == 'mse_loss' else F.l1_loss


    def forward(self, attention_head, binding_site):
        """
        attention_head: torch.Tensor
            The attention head from the transformer model. The shape should be (B, H, N)
        binding_site: torch.Tensor
            The binding site of the protein. The shape should be (B, N)
        """
        # pull out the first attention head
        attention_head = attention_head[:,0,:]


        # Calculate the loss
        loss = F.mse_loss(attention_head, binding_site)

        return loss

class PatternDecorrelationLoss(torch.nn.Module):
    def __init__(self):
        super(PatternDecorrelationLoss, self).__init__()

    def forward(self, attention_head):
        """
        attention_head: torch.Tensor
            The attention head from the transformer model. The shape should be (B, H, N)
        """
        # Calculate the loss
        loss = torch.linalg.matrix_norm(attention_head.mT @ attention_head - torch.eye(attention_head.shape[2]).to(attention_head), ord='fro').square().sum()

        return loss
