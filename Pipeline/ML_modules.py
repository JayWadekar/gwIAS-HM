# To use these modules, you need
# conda install pytorch cpuonly -c pytorch
# conda install lightning -c conda-forge
# pip install nflows

# To train the model, you could use gpus in pytorch
# but the models as of 08/24 can be easily be trained with just cpus
# in a very reasonable amount of time.

# The notebook associated with these modules is available at:
# pipeline_usage/22+HM/1.5Template_Prior.ipynb

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

def build_mlp(input_dim, hidden_dim, output_dim, layers, activation=nn.GELU()):
    """
    Create an multi-layer perceptron (MLP) from the configuration.
    """
    seq = [nn.Linear(input_dim, hidden_dim), activation]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation]
    seq += [nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*seq)

def get_flow(d_in=3, d_hidden=64, d_context=2, n_layers=4):
    """ 
    Instantiate a simple Masked Autoregressive normalizing flow.
    """

    base_dist = StandardNormal(shape=[d_in])

    transforms = []
    for _ in range(n_layers):
        transforms.append(ReversePermutation(features=d_in))
        transforms.append(MaskedAffineAutoregressiveTransform(features=d_in, 
                            hidden_features=d_hidden, context_features=d_context))
    transform = CompositeTransform(transforms)

    flow = Flow(transform, base_dist)
    return flow

class NeuralPosteriorEstimator(pl.LightningModule):
    """ 
    Simple neural posterior estimator class using a normalizing flow as the posterior density estimator.
    """
    def __init__(self, featurizer=None, d_context=2, d_in=3, d_hidden_flow=16, n_layers_flow=4,
                 d_hidden_featurizer=32, n_layers_featurizer=4):
        """
        :param featurizer: nn.Module
            The featurizer to use to transform its input to the context space.
        :param d_context: The dimension of the context space.
        :param d_in: The dimension of the input space.
        :param d_hidden_flow: The number of hidden units in the flow.
        :param n_layers_flow: The number of layers in the flow.
        :param d_hidden_featurizer: The number of hidden units in the featurizer.
        :param n_layers_featurizer: The number of layers in the featurizer.
        """
        super().__init__()
        if featurizer is None:
            self.featurizer = build_mlp(input_dim=d_context, 
                    hidden_dim=d_hidden_featurizer, output_dim=d_context,
                    layers=n_layers_featurizer)
        else:
            self.featurizer = featurizer
        self.flow = get_flow(d_in=d_in, d_hidden=d_hidden_flow,
                             d_context=d_context, n_layers=n_layers_flow)

    def forward(self, x):
        return self.featurizer(x)
    
    def loss(self, x, theta):
        context = self(x)
        return -self.flow.log_prob(inputs=theta, context=context)

    def training_step(self, batch, batch_idx):
        x, theta = batch
        loss = self.loss(x, theta).mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, theta = batch
        loss = self.loss(x, theta).mean()
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
    def generate_samples(self, calpha, num_samples=2048, set_seed=False):
        """
        Generate samples from of mode amplitude ratios (e.g., R33=A33/A22) 
        as a function of the input calpha.
        :param calpha: np.array
            The input calpha to generate samples for.
        :param num_samples: int
            The number of samples to generate.
        :param set_seed: bool
            Whether to set the seed for reproducibility.
        :return: [R33_samples, R44_samples, weight_samples]
                the weights correspond to the sensitive volume of the samples
        """
        calpha_norm = torch.FloatTensor(
                (calpha - self.calphas_mean) / self.calphas_std)
        context = self.featurizer(calpha_norm).unsqueeze(0)
        if set_seed:
            torch.manual_seed(0)
        mode_ratios_samples = self.flow.sample(num_samples=num_samples,
                        context=context).detach().numpy()[0]
        mode_ratios_samples = mode_ratios_samples * self.Rij_std + self.Rij_mean
        mask = np.any(mode_ratios_samples<0, axis=-1)
        mode_ratios_samples = mode_ratios_samples[~mask]
        return mode_ratios_samples
    
    @classmethod
    def load_from_path(cls, path):
        return torch.load(path)
    
class Template_Prior_NF(pl.LightningModule):
    """ 
    param:
    """
    def __init__(self, loss_threshold=None, d_in=2, d_hidden_flow=32, n_layers_flow=4):
        """
        :param loss_threshold: The threshold below which reject unphysical calphas
        :param d_in: The dimension of the input space.
        :param d_hidden_flow: The number of hidden units in the flow.
        :param n_layers_flow: The number of layers in the flow.
        self.log_bank_norm: Pre-stored variable for normalization factor of the bank
                         calculated using astrophysical prior samples
        self.calpha_reject_threshold: Pre-stored variable for threshold on
                                     log_prior for rejecting unphysical calphas
        """
        super().__init__()
        if loss_threshold is not None:
            self.loss_threshold = loss_threshold
        self.flow = get_flow(d_in=d_in, d_hidden=d_hidden_flow, n_layers=n_layers_flow,
                             d_context=None)

    def loss(self, calphas, weights=None):
        if weights is None:
            return -self.flow.log_prob(inputs=calphas)
        else:
            return -self.flow.log_prob(inputs=calphas) * weights

    def training_step(self, batch):
        calphas, weights = batch
        loss = self.loss(calphas, weights).mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        calphas, weights = batch
        loss = self.loss(calphas, weights).mean()
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
    def log_prior(self, calpha_array):
        """
        Compute the prior of the calpha templates.
        :param calpha_norm_array: array of calpha templates normalized
                                  using bank.calpha_transformer.transform(calphas)
        :return: Log of the prior density at the location of the template.
            For the probability of a template, use:
            np.exp(prior) * (delta_calpha**grid_ndims)
        """
        calpha_norm = torch.FloatTensor((calpha_array
                                - self.calphas_mean) / self.calphas_std)
        log_prior = -self.loss(calpha_norm).detach().numpy()
        log_prior += self.log_bank_norm
        return log_prior
    
    @classmethod
    def load_from_path(cls, path):
        return torch.load(path)