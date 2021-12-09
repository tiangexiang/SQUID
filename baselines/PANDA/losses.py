import torch
import torch.nn as nn

class CompactnessLoss(nn.Module):
    def __init__(self, center):
        super(CompactnessLoss, self).__init__()
        self.center = center

    def forward(self, inputs):
        m = inputs.size(1)
        variances = (inputs - self.center).norm(dim=1).pow(2) / m
        return variances.mean()


class EWCLoss(nn.Module):
    def __init__(self, frozen_model, fisher, lambda_ewc=1e4):
        super(EWCLoss, self).__init__()
        self.frozen_model = frozen_model
        self.fisher = fisher
        self.lambda_ewc = lambda_ewc

    def forward(self, cur_model):
        loss_reg = 0
        for (name, param), (_, param_old) in zip(cur_model.named_parameters(), self.frozen_model.named_parameters()):
            if 'fc' in name:
                continue
            loss_reg += torch.sum(self.fisher[name]*(param_old-param).pow(2))/2
        return self.lambda_ewc * loss_reg
