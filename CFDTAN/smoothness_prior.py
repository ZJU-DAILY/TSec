import torch
from difw import Cpab


def torch_dist_mat(centers):
    times = centers.shape

    centers_grid = centers.repeat(times[0], 1)
    dist_matrix = torch.abs(centers_grid - torch.transpose(centers_grid, 0, 1))
    return dist_matrix


def smoothness_norm(T: Cpab, theta, lambda_smooth=0.5, lambda_var=0.1, back_version=True, print_info=False):
    cov_cpa = T.covariance_cpa(length_scale=lambda_smooth, output_variance=lambda_var)
    precision_theta = torch.inverse(cov_cpa)
    smooth_norm = 0

    if back_version:
        theta = torch.squeeze(theta)
        if theta.ndim == 1:
            theta = torch.unsqueeze(theta, 0)
        theta_t = torch.transpose(theta, 0, 1)
        smooth_norm = torch.matmul(theta, torch.matmul(precision_theta, theta_t))
        smooth_norm = torch.mean(smooth_norm)
    else:
        assert (len(theta.shape) == 3)
        n_channels = theta.shape[0]
        theta_t = torch.transpose(theta, 1, 2)
        for idx in range(n_channels):
            idx_theta = theta[idx]
            idx_theta_t = theta_t[idx]
            temp_smooth_norm = torch.matmul(idx_theta, torch.matmul(precision_theta, idx_theta_t))
            temp_smooth_norm = torch.mean(temp_smooth_norm)
            smooth_norm += temp_smooth_norm
        smooth_norm /= n_channels

    return smooth_norm
