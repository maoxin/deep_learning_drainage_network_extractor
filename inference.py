from pathlib import Path
from argparse import ArgumentParser

import torch
from osgeo import gdal
import numpy as np
from tqdm import tqdm
import cv2

from models.dn_extractor import DrainageNetworkExtractor
from pytorch_lightning import Trainer
from data.nhdplus_hr_methods import DataHandler


DH = DataHandler()

def get_model(args):
    dict_args = vars(args)
    use_level_label = False
    if dict_args['use_model'].lower() == 'lhn_unet':
        use_level_label = True

    print("load model")
    model = DrainageNetworkExtractor(**dict_args)

    print("load state")
    state = get_ckpt_path(args.name)
    model.load_state_dict(state)
    model = model.eval()

    return model


def get_ckpt_path(name):
    if name == "ex_13":
        ckpt_path = Path(__file__).resolve().parent/"states/states_ex_13.ckpt"
    else:
        raise Exception("13 (U-Net + aspect features) is provided")

    print(f"Load state from {ckpt_path}")

    state = torch.load(str(ckpt_path), map_location='cpu')['state_dict']

    return state


def load_elev_tif(path, is_meter=False):
    elev_tif = gdal.Open(str(path), gdal.GA_ReadOnly)

    elev_tif_array = elev_tif.ReadAsArray().astype(np.float32)
    if is_meter:
        elev_tif_array *= 100

    return elev_tif, elev_tif_array


def infer(model, tif_path, is_meter=False):
    elev_tif, elev_tif_array = load_elev_tif(tif_path, is_meter)
    sample = DH.process(elev_tif_array)

    x_fdr, x_swnet, x_swnet_logits, d8 = model.infer_simple(sample)
    x_swnet_logits = x_swnet_logits[:, :, 1] + x_swnet_logits[:, :, 2]
    # x_swnet_logits[x_swnet == 0] = 0

    y_res, x_res = x_swnet.shape
    x_swnet_ = np.zeros((y_res + 2, x_res + 2), dtype='uint8')
    x_swnet_[1:-1, 1:-1] = x_swnet
    x_swnet_[(x_swnet_ == 1) | (x_swnet_ == 2)] = 255
    skeleton = cv2.ximgproc.thinning(x_swnet_)[1:-1, 1:-1]

    num_labels, labels_im = cv2.connectedComponents(skeleton)
    boarder_mask = np.zeros(skeleton.shape, dtype='bool')
    boarder_mask[:5] = True
    boarder_mask[-5:] = True
    boarder_mask[:, :5] = True
    boarder_mask[:, -5:] = True

    for l in range(1, num_labels + 1):
        l_indices = (labels_im == l)
        if not (l_indices & boarder_mask).any():
            labels_im[l_indices] = 0

    labels_im[labels_im != 0] = 255

    return x_swnet.detach().cpu().numpy(), x_swnet_logits.detach().cpu().numpy(),\
           x_fdr.detach().cpu().numpy(), labels_im, elev_tif


def save_results(x_swnet_logits, flowlines, swnet, fdr, elev_tif, save_dir, name):
    elev_proj = elev_tif.GetProjection()
    x_res = elev_tif.RasterXSize - 6
    y_res = elev_tif.RasterYSize - 6
    geo_transform = elev_tif.GetGeoTransform()

    rx = geo_transform[1]
    ry = geo_transform[5]
    geo_transform = list(geo_transform)
    geo_transform[0] += 3 * rx
    geo_transform[3] += 3 * ry
    geo_transform = tuple(geo_transform)

    target_ds_logits = gdal.GetDriverByName('GTiff').Create(str(Path(save_dir)/f"{name}_logits.tif"),
                                                            x_res, y_res, 1, gdal.GDT_Float32)
    target_ds_logits.GetRasterBand(1).WriteArray(x_swnet_logits)
    target_ds_logits.SetGeoTransform(geo_transform)
    target_ds_logits.SetProjection(elev_proj)
    target_ds_logits.FlushCache()
    target_ds_logits = None

    target_ds_flowlines = gdal.GetDriverByName('GTiff').Create(str(Path(save_dir) / f"{name}_flowline.tif"),
                                                               x_res, y_res, 1, gdal.GDT_Byte)
    target_ds_flowlines.GetRasterBand(1).WriteArray(flowlines)
    target_ds_flowlines.SetGeoTransform(geo_transform)
    target_ds_flowlines.SetProjection(elev_proj)
    target_ds_flowlines.FlushCache()
    target_ds_flowlines = None

    target_ds_waterbody_polygon = gdal.GetDriverByName('GTiff').Create(str(Path(save_dir) / f"{name}_waterbody_polygon.tif"),
                                                               x_res, y_res, 1, gdal.GDT_Byte)
    target_ds_waterbody_polygon.GetRasterBand(1).WriteArray(swnet)
    target_ds_waterbody_polygon.SetGeoTransform(geo_transform)
    target_ds_waterbody_polygon.SetProjection(elev_proj)
    target_ds_waterbody_polygon.FlushCache()
    target_ds_waterbody_polygon = None

    target_ds_fdr = gdal.GetDriverByName('GTiff').Create(str(Path(save_dir) / f"{name}_fdr.tif"),
                                                         x_res, y_res, 1, gdal.GDT_UInt16)
    target_ds_fdr.GetRasterBand(1).WriteArray(fdr)
    target_ds_fdr.SetGeoTransform(geo_transform)
    target_ds_fdr.SetProjection(elev_proj)
    target_ds_fdr.FlushCache()
    target_ds_fdr = None


def infer_and_save(model, tif_path, save_dir, is_meter=False):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    save_dir = Path(save_dir)
    tif_path = Path(tif_path)
    swnet, logits, fdr, flowlines, elev_tif = infer(model, tif_path, is_meter)
    swnet[swnet == 2] = 255
    swnet[swnet == 1] = 255

    name = tif_path.stem
    save_results(logits, flowlines, swnet, fdr, elev_tif, save_dir, name)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--default-root-dir', type=str, default="./")
    parser.add_argument('--log-dir', type=str, default="./")
    parser.add_argument('--name', type=str, default="ex0.args")
    parser.add_argument('--data-dir', type=str, default="./")
    parser.add_argument('--data-records-path', type=str, default="./records.json")
    parser.add_argument('--gradient-clip-val', type=float, default=0.5)
    parser.add_argument('--max-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.fromfile_prefix_chars = "@"

    # let the model add what it wants
    parser = DrainageNetworkExtractor.add_model_specific_args(parser)

    args = parser.parse_args()

    model = get_model(args)

    save_dir = Path("./results4samples").resolve()
    save_dir.mkdir(exist_ok=True)

    for elev_path in tqdm(list(Path("./samples").glob("*.tif"))):
        infer_and_save(model, elev_path, save_dir)

