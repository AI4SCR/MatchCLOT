import torch
import torch.nn.functional as F
from torch import nn


class BatchSwapNoise(nn.Module):
    """Swap Noise module"""

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mask = torch.rand(x.size()) > (1 - self.p)
            idx = torch.add(
                torch.arange(x.nelement()),
                (
                    torch.floor(torch.rand(x.size()) * x.size(0)).type(torch.LongTensor)
                    * (mask.type(torch.LongTensor) * x.size(1))
                ).view(-1),
            )
            idx[idx >= x.nelement()] = idx[idx >= x.nelement()] - x.nelement()
            return x.view(-1)[idx].view(x.size())
        else:
            return x


class Encoder(nn.Module):
    """Single modality encoder MLP with dropout and stochastic feature augmentation (SFA)
    https://openaccess.thecvf.com/content/ICCV2021/papers
    /Li_A_Simple_Feature_Augmentation_for_Domain_Generalization_ICCV_2021_paper.pdf"""

    def __init__(
        self,
        n_input,
        embedding_size,
        dropout_rates,
        dims_layers,
        swap_noise_ratio,
        noise_amount=0.0,
    ):
        super(Encoder, self).__init__()
        dropout = []
        layers = [nn.Linear(n_input, dims_layers[0])]

        for i in range(len(dims_layers) - 1):
            layers.append(nn.Linear(dims_layers[i], dims_layers[i + 1]))
        for i in range(len(dropout_rates)):
            dropout.append(nn.Dropout(p=dropout_rates[i]))

        layers.append(nn.Linear(dims_layers[-1], embedding_size))

        self.fc_list = nn.ModuleList(layers)
        print("dropout list", dropout)
        self.dropout_list = nn.ModuleList(dropout)
        self.noise_amount = noise_amount
        print("SFA with noise:", noise_amount)

    def forward(self, x):
        for i in range(len(self.fc_list) - 1):
            if i > 0 and self.training and self.noise_amount > 0:
                x = torch.mul(
                    x,
                    torch.ones_like(x)
                    + self.noise_amount * torch.randn_like(x, device=x.device),
                )
                x += self.noise_amount * torch.randn_like(x, device=x.device)
            x = F.elu(self.fc_list[i](x))
            if i < len(self.dropout_list):
                x = self.dropout_list[i](x)

        x = self.fc_list[-1](x)
        return x


class Modality_CLIP(nn.Module):
    """CLIP-inspired architecture"""

    def __init__(
        self,
        Encoder,
        layers_dims,
        dropout_rates,
        dim_mod1,
        dim_mod2,
        output_dim,
        T,
        swap_rate_1=0.0,
        swap_rate_2=0.0,
        noise_amount=0.0,
    ):
        super(Modality_CLIP, self).__init__()

        self.encoder_modality1 = Encoder(
            dim_mod1,
            output_dim,
            dropout_rates[0],
            layers_dims[0],
            swap_rate_1,
            noise_amount=noise_amount,
        )
        self.encoder_modality2 = Encoder(
            dim_mod2,
            output_dim,
            dropout_rates[1],
            layers_dims[1],
            swap_rate_2,
            noise_amount=noise_amount,
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * T)

    def forward(self, features_first, features_second):
        features_mod1 = self.encoder_modality1(features_first)
        features_mod2 = self.encoder_modality2(features_second)

        features_mod1 = features_mod1 / torch.norm(
            features_mod1, p=2, dim=-1, keepdim=True
        )
        features_mod2 = features_mod2 / torch.norm(
            features_mod2, p=2, dim=-1, keepdim=True
        )

        logit_scale = self.logit_scale.exp()

        logits = logit_scale * features_mod1 @ features_mod2.T

        return logits, features_mod1, features_mod2


def symmetric_npair_loss(logits, targets):
    """CLIP loss"""
    loss = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))
    return loss
