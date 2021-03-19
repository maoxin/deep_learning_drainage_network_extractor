from pathlib import Path
import json
import random
import pickle

import numpy as np
from scipy import signal
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch import nn
import networkx as nx
import richdem as rd


class NHDPlusHRDataSet(Dataset):
    def __init__(self, data_dir, data_records_path, split='train',
                 transform=None, use_level_label=False, use_coord=False,
                 use_d8=False, use_slope=False, use_curvature=False,
                 use_acc=False):
        self.data_dir = Path(data_dir)
        self.data_records_path = Path(data_records_path)
        self.split = split
        self.transform = transform
        self.use_level_label = use_level_label
        self.use_coord = use_coord
        self.use_d8 = use_d8
        self.use_slope = use_slope
        self.use_acc = use_acc
        self.use_curvature = use_curvature

        if self.use_coord:
            self.i_ar, self.j_ar = np.meshgrid(range(512), range(512), indexing='ij')
            self.i_ar = (self.i_ar - 255.5) / 255.5
            self.j_ar = (self.j_ar - 255.5) / 255.5
            self.coords = torch.stack([
                torch.FloatTensor(self.i_ar), torch.FloatTensor(self.j_ar)
            ], dim=0)

        if not self.data_records_path.exists():
            self.build_data_records()

        print("ds: load record")
        self.data_records = []
        with self.data_records_path.open('r') as f:
            for l in f:
                record = json.loads(l.strip())
                if self.split == 'all' or record['split'] == self.split:
                    self.data_records.append(record)

        # self.lhn_samplers = nn.ModuleList([nn.AvgPool2d(2 ** d, 2 ** d) for d in range(1, 5)])
        # self.lhn_mask_samplers = nn.ModuleList([nn.MaxPool2d(2 ** d, 2 ** d) for d in range(1, 5)])
        if self.use_level_label:
            self.lhn_sl_pcas = []
            for i in range(1, 5):
                pca_path = list(self.data_records_path.parent.glob(f"*fdr_swnet*level{i}.pickle"))[0]
                with pca_path.open("rb") as f:
                    self.lhn_sl_pcas.append(pickle.load(f))

    def __getitem__(self, idx):
        record = self.data_records[idx]
        elev_cm = np.load(str(self.data_dir/record['elev_cm_path'])).astype(np.float32)
        wbd = np.load(str(self.data_dir / record['wbdhu4_path'])).astype(np.int64)
        fdr = np.load(str(self.data_dir/record['fdr_path'])).astype(np.int64)
        swnet = np.load(str(self.data_dir/record['swnet_path'])).astype(np.int64)
        # (H + 2pad, W + 2pad)

        pad = record['pad']
        kernel_size = pad * 2 + 1
        elev_cm_tpi = self.get_tpi(elev_cm, kernel_size=kernel_size)
        # openness = self.get_openness(elev_cm, radius=pad)
        # roughness = self.get_roughness(elev_cm, kernel_size=kernel_size)
        # slope = self.get_slope(elev_cm, pad=pad)
        # too slow, so do not use extra info currently
        # x = torch.FloatTensor(np.concatenate([elev_cm_tpi, openness, roughness, slope], axis=2)).permute(2, 0, 1)
        # (12, H, W):
        #  [0] for tpi
        #  [1, 8] for openness, E, W, S, N, E, W, S, N
        #  [9] for roughness
        #  [10, 11] for slope, S, E
        # assert (~np.isnan(elev_cm_tpi)).all()
        x = torch.FloatTensor(elev_cm_tpi).permute(2, 0, 1)
        # (1, H, W)

        slope = None
        curvature = None
        if self.use_curvature and self.use_slope:
            slope, curvature = self.get_slope_curvature(elev_cm, pad=pad)
            slope = torch.FloatTensor(slope).permute(2, 0, 1)
            curvature = torch.FloatTensor(curvature).permute(2, 0, 1)
        elif self.use_slope:
            slope = torch.FloatTensor(self.get_slope_m(elev_cm, pad=pad)).permute(2, 0, 1)
        elif self.use_curvature:
            curvature = torch.FloatTensor(self.get_curvature(elev_cm, pad=pad)).permute(2, 0, 1)

        if self.use_d8:
            d8 = torch.LongTensor(self.get_d8(elev_cm, pad=pad)).unsqueeze(0)
            # (1, H, W)
        acc = None
        if self.use_acc:
            flow_accer = FlowAccumulation(d8)
            flow_accer.get_acc()
            acc = torch.FloatTensor(flow_accer.acc[pad:-pad, pad:-pad]).unsqueeze(0)

        mask = torch.LongTensor(wbd[pad:-pad, pad:-pad])
        # (H, W)

        fdr = fdr[pad:-pad, pad:-pad]
        fdr[fdr != 0] = np.log2(fdr[fdr != 0]).astype(np.int64) + 1
        # assert (~np.isnan(fdr)).all()
        # 0: flow ends (sink)
        # 1 -> 8: E, SE, S, SW, W, NW, N, NE
        swnet = swnet[pad:-pad, pad:-pad] % 3
        # assert (~np.isnan(swnet)).all()
        # 0: cell not on flow network
        # 1: network flow cell
        # 2: waterbody cell
        y = torch.LongTensor(np.stack([fdr, swnet], axis=2)).permute(2, 0, 1)
        # (2, H, W), 1 for fdr, and 2 for swnet

        # fdr_oh = np.zeros((*fdr.shape, 9)).astype(np.float32)
        # for i in range(9):
        #     fdr_oh[:, :, i][fdr == i] = 1
        # swnet_oh = np.zeros((*swnet.shape, 3)).astype(np.float32)
        # for i in range(3):
        #     swnet_oh[:, :, i][swnet == i] = 1
        # for i, lhn_sampler in enumerate(self.lhn_samplers):
        #     sl_pca = self.lhn_sl_pcas[i]
        #     # use_cmp_before = np.where(sl_pca.explained_variance_ratio_.cumsum() >= 0.95)[0][0] + 1
        #     use_cmp_before = 8
        #     size = int(512 / 2**(i + 1))
        #     y_level = lhn_sampler(
        #         torch.FloatTensor(np.concatenate([fdr_oh, swnet_oh], axis=2)).permute(2, 0, 1).unsqueeze(0)
        #     ).squeeze(0).numpy().transpose(1, 2, 0).reshape(size * size, -1)
        #     y_level = sl_pca.transform(y_level)[:, :use_cmp_before].reshape(size, size, -1)
        #     y_levels.append(torch.FloatTensor(y_level).permute(2, 0, 1))

        if self.transform is not None:
            x, mask, y, slope, acc, curvature = self.transform(x, mask, y, slope=slope, acc=acc, curvature=curvature)

        if self.use_coord:
            x = torch.cat([x, self.coords], dim=0)

        if self.split != 'train' or not self.use_level_label:
            sample = {
                'x': x,
                'mask': mask,
                'y': y,
            }

            if self.use_d8:
                sample['d8'] = d8

            if self.use_slope:
                sample['slope'] = slope

            if self.use_curvature:
                sample['curvature'] = curvature

            if self.use_acc:
                sample['acc'] = acc

            return sample


        fdr_levels = []
        for i in range(1, 5):
            fdr_level_name = '_'.join([*record['fdr_path'].split('.')[0].split('_'), f'level{i}']) + '.npy'
            fdr_levels.append(np.load(str(self.data_dir / fdr_level_name)))
        swnet_levels = []
        for i in range(1, 5):
            swnet_level_name = '_'.join([*record['swnet_path'].split('.')[0].split('_'), f'level{i}']) + '.npy'
            swnet_levels.append(np.load(str(self.data_dir / swnet_level_name)))
        wbd_levels = []
        for i in range(1, 5):
            wbd_level_name = '_'.join([*record['wbdhu4_path'].split('.')[0].split('_'), f'level{i}']) + '.npy'
            wbd_levels.append(np.load(str(self.data_dir / wbd_level_name)))

        y_levels = []
        for i, (fdr_level, swnet_level) in enumerate(zip(fdr_levels, swnet_levels)):
            sl_pca = self.lhn_sl_pcas[i]
            use_cmp_before = int(np.where(sl_pca.explained_variance_ratio_.cumsum() >= 0.95)[0][0] + 1)

            size = int(512 / 2 ** (i + 1))
            y_level = np.concatenate([fdr_level, swnet_level], axis=2).reshape(size * size, -1)
            y_levels.append(torch.FloatTensor(
                sl_pca.transform(y_level)[:, :use_cmp_before].reshape(size, size, -1)).permute(2, 0, 1)
            )
            
        mask_levels = []
        for wbd_level in wbd_levels:
            mask_levels.append(torch.LongTensor(wbd_level))

        sample = {
            'x': x,
            'mask': mask,
            'y': y,
            'y_l1': y_levels[0],
            'y_l2': y_levels[1],
            'y_l3': y_levels[2],
            'y_l4': y_levels[3],
            'mask_l1': mask_levels[0],
            'mask_l2': mask_levels[1],
            'mask_l3': mask_levels[2],
            'mask_l4': mask_levels[3],
        }

        if self.use_d8:
            sample['d8'] = d8

        if self.use_slope:
            sample['slope'] = slope

        if self.use_curvature:
            sample['curvature'] = curvature

        if self.use_acc:
            sample['acc'] = acc

        return sample

    def build_data_records(self):
        records = []
        for file_name in tqdm(list(self.data_dir.glob("*elev_cm*.npy")), desc='build data records'):
            file_name = file_name.stem
            file_name = file_name.split('_')
            HU4 = file_name[1]
            i_start = int(file_name[2][1:])
            j_start = int(file_name[3][1:])
            crop = int(file_name[-2][4:])
            pad = int(file_name[-1][3:])
            head = "_".join(file_name[:4])
            tail = "_".join(file_name[-2:]) + ".npy"
            record = {
                'HU4': HU4, 'i_start': i_start, 'j_start': j_start,
                'crop': crop, 'pad': pad,
                'elev_cm_path': f"{head}_elev_cm_{tail}",
                'wbdhu4_path': f"{head}_wbd_{tail}",
                'fdr_path': f"{head}_fdr_{tail}",
                'swnet_path': f"{head}_swnet_{tail}",
            }

            elev_cm = np.load(str(self.data_dir/record['elev_cm_path']))
            record['elev_mean'] = float(elev_cm.mean())
            record['elev_std'] = float(elev_cm.std())
            records.append(record)

        random.seed(4399)
        inds = list(range(len(records)))
        num_train = int(len(records) * 0.7)
        num_val = int(len(records) * 0.2)
        train_inds = random.sample(inds, num_train)
        inds = [i for i in inds if i not in train_inds]
        val_inds = random.sample(inds, num_val)
        test_inds = [i for i in inds if i not in val_inds]

        for i in train_inds:
            records[i]['split'] = 'train'
        for i in val_inds:
            records[i]['split'] = 'val'
        for i in test_inds:
            records[i]['split'] = 'test'

        with self.data_records_path.open("w") as f:
            for r in records:
                f.write(f"{json.dumps(r)}\n")


    def get_dataset_statistics(self):
        means = []
        stds = []
        slope_means = []
        slope_stds = []
        curvature_means = []
        curvature_stds = []
        # acc_means = []
        # acc_stds = []
        for idx in tqdm(range(self.__len__()), desc='calculate statistics'):
            sample = self.__getitem__(idx)
            x = sample['x']
            means.append(x.flatten(start_dim=1).mean(dim=1))
            stds.append(x.flatten(start_dim=1).std(dim=1))

            slope = sample['slope']
            slope_means.append(slope.flatten(start_dim=1).mean(dim=1))
            slope_stds.append(slope.flatten(start_dim=1).std(dim=1))

            curvature = sample['curvature']
            curvature_means.append(curvature.flatten(start_dim=1).mean(dim=1))
            curvature_stds.append(curvature.flatten(start_dim=1).std(dim=1))

            # acc = sample['acc']
            # acc_means.append(acc.flatten(start_dim=1).mean(dim=1))
            # acc_stds.append(acc.flatten(start_dim=1).std(dim=1))

        means = torch.stack(means, dim=0).mean(dim=0)
        stds = torch.stack(stds, dim=0).mean(dim=0)
        slope_means = torch.stack(slope_means, dim=0).mean(dim=0)
        slope_stds = torch.stack(slope_stds, dim=0).mean(dim=0)
        curvature_means = torch.stack(curvature_means, dim=0).mean(dim=0)
        curvature_stds = torch.stack(curvature_stds, dim=0).mean(dim=0)
        # acc_means = torch.stack(acc_means, dim=0).mean(dim=0)
        # acc_stds = torch.stack(acc_stds, dim=0).mean(0)

        return {
            'mean': means,
            'std': stds,
            'slope_mean': slope_means,
            'slope_std': slope_stds,
            'curvature_mean': curvature_means,
            'curvature_std': curvature_stds,
            # 'acc_mean': acc_means,
            # 'acc_std': acc_stds,
        }


    def __len__(self):
        return len(self.data_records)
        # return 100

    def get_tpi(self, dem_img, kernel_size=7):
        """

        :param dem_img: numpy_array, (H + 2pad, W + 2pad)
        :param kernel_size: int
        :return: (H, W, 1)
        """

        kernel = np.zeros((kernel_size, kernel_size))
        num_cell = kernel_size ** 2
        kernel.fill(-1)
        #
        center = (kernel_size - 1) // 2
        kernel[center, center] = num_cell - 1

        # tpi = signal.convolve2d(dem_img, kernel, boundary='symm', mode='same')
        tpi = signal.convolve2d(dem_img, kernel, mode='valid')
        tpi = tpi / num_cell

        return tpi[:, :, None]

    def get_openness(self, dem_img, radius=3):
        """

        :param dem_img: (H + 2 pad, W + 2 pad)
        :param radius: int
        :return: (H, W, 8)
        """

        # E
        dem_img_ = dem_img[radius:-radius, radius:]
        p0 = dem_img_[:, :-radius][:, :, None]
        # (H, W, 1)
        p1 = np.concatenate([
            dem_img_[:, j:dem_img_.shape[1] - radius + j][:, :, None] for j in range(1, radius + 1)], axis=2)
        # (H, W, r)
        ys = p1 - p0
        xs = np.arange(1, 1 + radius)[None, None, :]
        # (1, 1, r)
        angles = np.arctan2(ys, xs)
        # (H, W, r)
        alpha_E = np.pi / 2 - angles.max(axis=2)
        # (H, W)
        beta_E = angles.min(axis=2) + np.pi / 2

        # W
        dem_img_ = dem_img[radius:-radius, :-radius]
        p0 = dem_img_[:, radius:][:, :, None]
        p1 = np.concatenate([
            dem_img_[:, radius - j: dem_img_.shape[1] - j][:, :, None] for j in range(1, radius + 1)], axis=2)
        ys = p1 - p0
        xs = np.arange(1, 1 + radius)[None, None, :]
        angles = np.arctan2(ys, xs)
        alpha_W = np.pi / 2 - angles.max(axis=2)
        beta_W = angles.min(axis=2) + np.pi / 2

        # S
        dem_img_ = dem_img[radius:, radius:-radius]
        p0 = dem_img_[:-radius, :][:, :, None]
        p1 = np.concatenate([
            dem_img_[i:dem_img_.shape[0] - radius + i, :][:, :, None] for i in range(1, radius + 1)], axis=2)
        ys = p1 - p0
        xs = np.arange(1, 1 + radius)[None, None, :]
        angles = np.arctan2(ys, xs)
        alpha_S = np.pi / 2 - angles.max(axis=2)
        beta_S = angles.min(axis=2) + np.pi / 2

        # N
        dem_img_ = dem_img[:-radius, radius:-radius]
        p0 = dem_img_[radius:, :][:, :, None]
        p1 = np.concatenate([
            dem_img_[radius - i: dem_img_.shape[0] - i, :][:, :, None] for i in range(1, radius + 1)], axis=2)
        ys = p1 - p0
        xs = np.arange(1, 1 + radius)[None, None, :]
        angles = np.arctan2(ys, xs)
        alpha_N = np.pi / 2 - angles.max(axis=2)
        beta_N = angles.min(axis=2) + np.pi / 2

        openness = np.stack([
            alpha_E, alpha_W, alpha_S, alpha_N,
            beta_E, beta_W, beta_S, beta_N,
        ], axis=2)

        return openness

    def get_roughness(self, dem_img, kernel_size=7):
        """

        :param dem_img: (H + 2 pad, W + 2 pad)
        :param radius: int
        :return: (H, W, 1)
        """

        pad = (kernel_size - 1) // 2

        dem_img_ = []
        for i in range(-pad, pad + 1):
            for j in range(-pad, pad + 1):
                dem_img_.append(
                    dem_img[pad + i:dem_img.shape[0] - pad + i, pad + j:dem_img.shape[1] - pad + j][:, :, None])
        dem_img_ = np.concatenate(dem_img_, axis=2)
        # (H, W, kernel_size**2)

        roughness = np.std(dem_img_, axis=2)

        return roughness[:, :, None]

    def get_slope(self, dem_img, pad=3):
        """
        return slope in two directions

        :param dem_img: (H + 2 pad, W + 2 pad)
        :param radius: int
        :return: (H, W, 2)
        """

        slope_i, slope_j = np.gradient(dem_img)
        slope = np.concatenate([slope_i[:, :, None], slope_j[:, :, None]], axis=2)
        return slope[pad:-pad, pad:-pad]

    def get_slope_m(self, dem_img, pad=3):
        """
        return the magnitude of the slope
        :param dem_img: (H + 2 pad, W + 2 pad)
        :param pad: int
        :return: (H, W, 1)
        """
        slope_i, slope_j = np.gradient(dem_img)
        slope = np.sqrt(slope_i**2 + slope_j**2)[:, :, None]

        return slope[pad:-pad, pad:-pad]

    def get_curvature(self, dem_img, pad=3, use_geometric=True):
        """
        return the curvature

        :param dem_img: (H + 2 pad, W + 2 pad)
        :param pad: int
        :param use_geometric: bool, use geometric curvature, or the laplacian curvature
        :return: (H, W, 1)
        """

        slope_i, slope_j = np.gradient(dem_img)
        slope = np.sqrt(slope_i ** 2 + slope_j ** 2)

        if use_geometric:
            slope_i = np.divide(slope_i, slope)
            slope_j = np.divide(slope_j, slope)

        grad_grad_i, _ = np.gradient(slope_i)
        _, grad_grad_j = np.gradient(slope_j)

        curvature = grad_grad_i + grad_grad_j
        curvature[np.isnan(curvature)] = 0

        return curvature[pad:-pad, pad:-pad, None]

    def get_slope_curvature(self, dem_img, pad=3, use_geometric=True):
        """
        return the slope magnitude and curvature

        :param dem_img: (H + 2 pad, W + 2 pad)
        :param pad: int
        :param use_geometric: bool, use geometric curvature, or the laplacian curvature
        :return: ((H, W, 1), (H, W, 1))
        """

        slope_i, slope_j = np.gradient(dem_img)
        slope = np.sqrt(slope_i ** 2 + slope_j ** 2)

        if use_geometric:
            slope_i = np.divide(slope_i, slope)
            slope_j = np.divide(slope_j, slope)

        grad_grad_i, _ = np.gradient(slope_i)
        _, grad_grad_j = np.gradient(slope_j)

        curvature = grad_grad_i + grad_grad_j
        curvature[np.isnan(curvature)] = 0

        return slope[pad: -pad, pad:-pad, None], curvature[pad:-pad, pad:-pad, None]

    def get_d8(self, dem_img, pad=3):
        if pad == 0:
            dem_img = np.pad(dem_img, 1, 'constant', constant_values=np.inf)
        elif pad != 1:
            dem_img = dem_img[pad-1: -(pad-1), pad-1: -(pad-1)]

        delta_x = []
        dem_img_ = []
        ijs = [
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
            (-1, 0),
            (-1, 1),
        ]

        for i, j in ijs:
            if abs(i) + abs(j) == 2:
                delta_x.append(2 ** 0.5)
            else:
                delta_x.append(1)

            dem_img_.append(
                dem_img[1 + i:dem_img.shape[0] - 1 + i, 1 + j:dem_img.shape[1] - 1 + j][:, :, None]
            )

        dem_img_ = np.concatenate(dem_img_, axis=2)
        # (H, W, 8)
        delta_x = np.array(delta_x)
        slope = (dem_img[1:-1, 1:-1][:, :, None] - dem_img_) / delta_x
        # (H, W, 8)
        d8 = slope.argmax(axis=2) + 1
        d8[slope.max(axis=2) < 0] = 0
        # (H, W)

        return d8

    def get_d8_without_depression(self, dem_img, pad=3):
        dem_rd = rd.rdarray(dem_img, no_data=-9999)
        dem_filled_rd = np.array(rd.FillDepressions(dem_rd, in_place=False))
        d8_without_depression = self.get_d8(dem_filled_rd, pad=pad)

        return d8_without_depression

    def get_vectors_from_d8(self, d8):
        d8 = d8.squeeze(0)

        i_s, j_s = torch.meshgrid(torch.arange(d8.shape[0]), torch.arange(d8.shape[1]))
        delta_ij = [
            torch.tensor([0, 1]),
            torch.tensor([1, 1]),
            torch.tensor([1, 0]),
            torch.tensor([1, -1]),
            torch.tensor([0, -1]),
            torch.tensor([-1, -1]),
            torch.tensor([-1, 0]),
            torch.tensor([-1, 1]),
        ]

        edges = []
        for d in range(1, 9):
            indices_d = (d8 == d)
            ijs_start_d = torch.stack([i_s[indices_d], j_s[indices_d]], dim=1).unsqueeze(1)
            ijs_end_d = ijs_start_d + delta_ij[d - 1]

            edges.append(torch.cat([ijs_start_d, ijs_end_d], dim=1))

        edges = torch.cat(edges, dim=0)

        filter = (0 <= edges[:, 1, 0]) & (edges[:, 1, 0] < d8.shape[0]) & \
                 (0 <= edges[:, 1, 1]) & (edges[:, 1, 1] < d8.shape[1])
        edges = edges[filter].tolist()
        edges = [[tuple(i) for i in pair] for pair in edges]

        DG = nx.DiGraph()
        DG.add_edges_from(edges)

        return DG


class FlowAccumulation:
    def __init__(self, d8):
        """

        :param d8: (1, H, W)
        """

        self.d8 = d8.squeeze(0).numpy()
        self.H, self.W = self.d8.shape
        self.lock = np.zeros_like(self.d8, dtype=np.bool)
        self.acc = np.full_like(self.d8, -1, dtype=np.int16)

        self.possible_predecessors = {
            (0, 1): 5,
            (1, 1): 6,
            (1, 0): 7,
            (1, -1): 8,
            (0, -1): 1,
            (-1, -1): 2,
            (-1, 0): 3,
            (-1, 1): 4,
        }

    def get_acc(self):
        for i in range(self.d8.shape[0]):
            for j in range(self.d8.shape[1]):
                if self.lock[i, j]:
                    continue
                self.get_acc_step(i, j)

    def get_acc_step(self, i, j):
        """
        get the acc for a point (i, j)

        :param i: int
        :param j: int
        :return:
        """

        self.lock[i, j] = True
        if self.acc[i, j] != -1:
            return self.acc[i, j]

        predecessors = self.get_predecessors(i, j)
        if len(predecessors) == 0:
            acc = 1
        else:
            acc = sum(self.get_acc_step(*prdc) for prdc in predecessors)
        self.acc[i, j] = acc
        return acc


    def get_predecessors(self, i, j):
        """
        get the predecessors of a point (i, j), which is inside the eight neighbors of (i, j), and is not locked

        :param i: int
        :param j: int
        :return:
        """

        predecessors = []
        for delta_ij in self.possible_predecessors:
            i_ = i + delta_ij[0]
            j_ = j + delta_ij[1]

            if (0 <= i_ < self.H) and (0 <= j_ < self.W):
                if (self.d8[i_, j_] == self.possible_predecessors[delta_ij]) and not self.lock[i_, j_]:
                    predecessors.append((i_, j_))

        return predecessors



class Transformer:
    def __init__(self, train=True):
        self.train = train

        try:
            stats = torch.load(str(Path(__file__).resolve().parent/"nhdplus_statistics_with_slope_curvature.pt"))
            self.mu = stats['mean']
            self.sigma = stats['std']
            self.mu_slope = stats['slope_mean']
            self.sigma_slope = stats['slope_std']
            self.mu_curvature = stats['curvature_mean']
            self.sigma_curvature = stats['curvature_std']
            # self.mu_acc = stats['acc_mean']
            # self.sigma_acc = stats['acc_std']


        except FileNotFoundError:
            raise Exception("'nhdplus_statistics.pt' cannot be found. Please run module 'data/nhdplus_hr.py' to generate")

    def __call__(self, x, mask, y, slope=None, curvature=None, acc=None):
        if self.train:
            pass
        x = (x - self.mu) / self.sigma

        if slope is not None:
            slope = (slope - self.mu_slope) / self.sigma_slope
        if curvature is not None:
            curvature = (curvature - self.mu_curvature) / self.sigma_curvature
        # if acc is not None:
        #     acc = (acc - self.mu_acc) / self.sigma_acc

        return (x, mask, y, slope, acc, curvature)


if __name__ == "__main__":
    RootDir = Path("/home/tan")
    DataRoot = RootDir / "mx_dataset/drainage_network_v2"
    DataDir = DataRoot / "samples"
    DataRecordsPath = DataRoot / "records.json"
    ds = NHDPlusHRDataSet(data_dir=DataDir, data_records_path=DataRecordsPath, use_slope=True,
                          use_curvature=True)

    # statistics = ds.get_dataset_statistics()
    # stats_path = str(Path(__file__).resolve().parent/"nhdplus_statistics_with_slope_curvature.pt")
    # torch.save(statistics, stats_path)

    sample = ds[0]
