import torch
from torch import nn


class KrigingLoss(nn.Module):
    def __init__(self, nlags=6, sample_size=256):
        """

        :param nlags: int, optional
        Number of averaging bins for the semivariogram. Default is 6.
        :param sample_size: int, optional
        Number of points used to calculate semivariance (256 for 16 x 16 grids)
        """

        super(KrigingLoss, self).__init__()
        self.nlags = nlags
        self.sample_size = sample_size

    def forward(self, x0, ind, x1):
        """

        :param x0: feature map in L-1 level (B, C, H, W)
        :param ind: indices in max pooling  (B, C, H, W)
        :param x1: feature map in L level (B, C, 2H, 2W)
        :return:
        """

        x0 = x0.detach()

        device = x0.get_device()
        H1, W1 = x1.shape[-2:]
        i_grid, j_grid = torch.meshgrid(torch.arange(H1, dtype=torch.float32).to(device),
                                        torch.arange(W1, dtype=torch.float32).to(device))
        i_grid = i_grid.expand_as(x1)
        j_grid = j_grid.expand_as(x1)

        # for x0
        x0 = x0.flatten(start_dim=2)
        ind = ind.flatten(start_dim=2)
        # (B, C, HW)
        perm_ind = torch.randperm(ind.size(2))

        x0 = x0[:, :, perm_ind[:self.sample_size]]
        ind = ind[:, :, perm_ind[:self.sample_size]]

        i_grid_used = self.retrieve_elements_from_indices(i_grid, ind)
        j_grid_used = self.retrieve_elements_from_indices(j_grid, ind)
        # (B, C, sample_size)

        d = torch.sqrt((i_grid_used.unsqueeze(3) - i_grid_used.unsqueeze(2))**2 +
                       (j_grid_used.unsqueeze(3) - j_grid_used.unsqueeze(2))**2).flatten(start_dim=2)
        # distance
        g = (0.5 * (x0.unsqueeze(3) - x0.unsqueeze(2)) ** 2).flatten(start_dim=2)
        # semivariance
        # (B, C, sample_size**2)

        dmax = d.max()
        dmin = d.min()
        dd = (dmax - dmin) / self.nlags
        bins = [dmin + n * dd for n in range(self.nlags)]
        dmax += 0.001
        bins.append(dmax)

        semivariance0 = torch.zeros([*x0.shape[:2], self.nlags + 2]).to(device)
        # (B, C, nlags + 2) the 2 more for outside

        for n in range(self.nlags):
            mask = torch.where((d >= bins[n]) & (d < bins[n + 1]),
                               torch.tensor([1.]).to(device),
                               torch.tensor([0.]).to(device))
            masked_s = torch.where((d >= bins[n]) & (d < bins[n + 1]),
                                   g, torch.tensor([0.]).to(device))

            semivariance0[:, :, n + 1] = masked_s.sum(dim=2) / (mask.sum(dim=2) + 1e-7)

            semivariance0[:, :, n+1][mask.sum(dim=2) == 0] = 0

        # for x1
        i_grid = i_grid.flatten(start_dim=2)
        j_grid = j_grid.flatten(start_dim=2)
        x1 = x1.flatten(start_dim=2)
        # (B, C, 4HW)
        perm_ind = torch.randperm(x1.size(2))

        i_grid = i_grid[:, :, perm_ind[:self.sample_size]]
        j_grid = j_grid[:, :, perm_ind[:self.sample_size]]
        x1 = x1[:, :, perm_ind[:self.sample_size]]

        d = torch.sqrt((i_grid.unsqueeze(3) - i_grid.unsqueeze(2)) ** 2 +
                       (j_grid.unsqueeze(3) - j_grid.unsqueeze(2)) ** 2).flatten(start_dim=2)
        # distance
        g = (0.5 * (x1.unsqueeze(3) - x1.unsqueeze(2)) ** 2).flatten(start_dim=2)
        # semivariance
        # (B, C, sample_size**2)

        semivariance1 = torch.zeros([*x1.shape[:2], self.nlags + 2]).to(device)
        # (B, C, nlags + 2) the 2 more for outside

        for n in range(self.nlags):
            mask = torch.where((d >= bins[n]) & (d < bins[n + 1]),
                               torch.tensor([1.]).to(device),
                               torch.tensor([0.]).to(device))
            masked_s = torch.where((d >= bins[n]) & (d < bins[n + 1]),
                                   g, torch.tensor([0.]).to(device))

            semivariance1[:, :, n + 1] = masked_s.sum(dim=2) / (mask.sum(dim=2) + 1e-7)

            semivariance1[:, :, n + 1][mask.sum(dim=2) == 0] = 0

        # left side
        mask = torch.where(d < bins[0],
                           torch.tensor([1.]).to(device),
                           torch.tensor([0.]).to(device))
        masked_s = torch.where(d < bins[0],
                               g, torch.tensor([0.]).to(device))

        semivariance1[:, :, 0] = masked_s.sum(dim=2) / (mask.sum(dim=2) + 1e-7)

        semivariance1[:, :, 0][mask.sum(dim=2) == 0] = 0

        # right side
        mask = torch.where(d >= bins[-1],
                           torch.tensor([1.]).to(device),
                           torch.tensor([0.]).to(device))
        masked_s = torch.where(d >= bins[-1],
                               g, torch.tensor([0.]).to(device))

        semivariance1[:, :, -1] = masked_s.sum(dim=2) / (mask.sum(dim=2) + 1e-7)

        semivariance1[:, :, -1][mask.sum(dim=2) == 0] = 0

        loss = ((semivariance1 - semivariance0)**2).mean()

        return loss

    def retrieve_elements_from_indices(self, tensor, indices):
        flattened_tensor = tensor.flatten(start_dim=2)
        output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
        return output


if __name__ == "__main__":
    import torch.nn.functional as F

    x1 = torch.randn(6, 64, 256, 256)
    x0, ind = F.max_pool2d_with_indices(x1, 2)

    kl = KrigingLoss()

    loss = kl(x0, ind, x1)
