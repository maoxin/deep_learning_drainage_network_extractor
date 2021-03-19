# Deep learning-enhanced extraction of drainage networks from digital elevation models

This is a deep learning-enhanced framework for drainage network extraction. This framework can predict flow
 directions, flowlines and waterbody polygons simultaneously, with digital elevation
 models (DEMs) as the input.
 
## Software required
* GDAL 2.2.3
* Opencv 3.2.0
* Python 3.7.9
    * pytorch 1.4.0
    * torchvision 0.5.0
    * pytorch-lightning 1.0.6
    * gdal
    * numpy
    * opencv-python
    * opencv-contrib-python
    * scipy
    * tqdm
    * networkx
    * richdem

Before using this framework, please also download the [states](https://hkustconnect-my.sharepoint.com/:u:/g/personal/xmaoac_connect_ust_hk/EZz98s0F-MFEl5XfwbukkLUBEjrePZ868lbz81K5xksfPQ?e=hoVBwR) of the deep learning models used in this framework, and decompress the "states.zip" file followed by putting the "states" directory in the project root directory.

### Usage

The files of the digital elevation models (\*\_elev\_cm.tif) used in our paper are provided in the "samples
" directory. By running the following command, the corresponding flow directions (\*\_fdr.tif), flowlines (\*\_flowline
.tif) and waterbody polygons (\*\_waterbody\_polygon.tif) files are produced in the "results4samples" directory.

```python
python inference.py @ex_configs/ex_13
```

The tif files of the digital elevation models can also be provided by users in the "samples" directory, but please
 ensure that the
 tif
 files
 contain no "No data value" and the elevation unit is centimeter.

 