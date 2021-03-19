from pathlib import Path

import torch

from data.nhdplus_hr import NHDPlusHRDataSet

class DataHandler(NHDPlusHRDataSet):
    def __init__(self):
        self.use_d8 = True
        self.use_level_label = False
        self.use_coord = False
        self.use_d8 = False
        self.use_slope = False
        self.use_acc = False
        self.use_curvature = False

        self.stats = torch.load(str(Path(__file__).resolve().parent/"nhdplus_statistics_with_slope_curvature.pt"))
        self.mu = self.stats['mean']
        self.sigma = self.stats['std']

    def process(self, elev_cm):
        pad = 3
        kernel_size = pad * 2 + 1
        elev_cm_tpi = self.get_tpi(elev_cm, kernel_size=kernel_size)
        x = torch.FloatTensor(elev_cm_tpi).permute(2, 0, 1).unsqueeze(1)
        x = (x - self.mu) / self.sigma

        d8 = torch.LongTensor(self.get_d8(elev_cm, pad=pad)).unsqueeze(0)

        return {
            'x': x,
            'd8': d8,
        }
