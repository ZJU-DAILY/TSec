import argparse

import difw
import torch

from CFDTAN.smoothness_prior import smoothness_norm
from utils.utils_func import CFArgs


def alignment_loss(x_transformed, labels, thetas, n_channels, cf_args: CFArgs):
    # torch.autograd.set_detect_anomaly(True)
    loss = 0
    align_loss = 0
    prior_loss = 0
    n_classes = labels.unique()
    tri_size = n_channels * (n_channels - 1) / 2
    # batch_size = x_transformed.shape[0]
    for i in n_classes:
        x_within_class = x_transformed[labels == i]
        class_size = x_within_class.shape[0]
        if n_channels == 1:
            loss = loss + x_within_class.var(dim=0, unbiased=False).mean()
        else:
            per_channel_loss = x_within_class.var(dim=0, unbiased=False).mean(dim=1)
            per_channel_loss = per_channel_loss.mean()
            loss = loss + per_channel_loss

            if not cf_args.back_version:
                cov_loss = 0
                # x_within_class = torch.transpose(x_within_class, 1, 2)
                for idx in range(class_size):
                    single_x_within_class = x_within_class[idx]
                    corr = torch.corrcoef(single_x_within_class)
                    corr = 1 - torch.abs(corr)
                    corr = torch.triu(corr, diagonal=1)
                    cov_loss = cov_loss + torch.sum(corr) / tri_size
                loss = loss + cov_loss

    loss /= len(n_classes)

    if cf_args.smoothness_prior:
        for theta in thetas:
            prior_loss = prior_loss + 0.1 * smoothness_norm(cf_args.T, theta, cf_args.lambda_smooth, cf_args.lambda_var,
                                                            back_version=cf_args.back_version, print_info=False)
            loss = loss + prior_loss

        return loss


def alignment_loss_T(x_transformed, labels, thetas, n_channels, cf_args: argparse.Namespace, T: difw.Cpab):
    loss = 0
    align_loss = 0
    prior_loss = 0
    n_classes = labels.unique()
    tri_size = n_channels * (n_channels - 1) / 2
    # batch_size = x_transformed.shape[0]
    for i in n_classes:
        x_within_class = x_transformed[labels == i]
        class_size = x_within_class.shape[0]
        if n_channels == 1:
            loss = loss + x_within_class.var(dim=0, unbiased=False).mean()
        else:
            per_channel_loss = x_within_class.var(dim=0, unbiased=False).mean(dim=1)
            per_channel_loss = per_channel_loss.mean()
            loss = loss + per_channel_loss

            if not cf_args.back_version:
                cov_loss = 0
                # x_within_class = torch.transpose(x_within_class, 1, 2)
                for idx in range(class_size):
                    single_x_within_class = x_within_class[idx]
                    corr = torch.corrcoef(single_x_within_class)
                    corr = 1 - torch.abs(corr)
                    corr = torch.triu(corr, diagonal=1)
                    cov_loss = cov_loss + torch.sum(corr) / tri_size
                loss = loss + cov_loss

    loss /= len(n_classes)

    if cf_args.smoothness_prior:
        for theta in thetas:
            # if theta.shape[0] == 1:
            #     print(thetas)
            prior_loss = prior_loss + 0.1 * smoothness_norm(T, theta, cf_args.lambda_smooth, cf_args.lambda_var,
                                                            back_version=cf_args.back_version, print_info=False)
            loss = loss + prior_loss

        return loss
