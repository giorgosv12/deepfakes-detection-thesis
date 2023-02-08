# Dataset Preprocessing
***

In the initial stage of training the Deepfakes Detection Models, preprocessing of the videos in the dataset is 
performed. The video frames are transformed into the pose, expression, shape, and texture parameters used by the [FLAME 
model](https://github.com/soubhiksanyal/FLAME_PyTorch), which utilizes the reconstruction of a 3DMM from a single image 
proposed by [DECA](https://github.com/yfeng95/DECA). Additionally, features proposed for the detection of deepfakes, as 
described in chapter 3.1 of the thesis, are calculated for each frame. A single master.npz file is created for each 
video, containing all the aforementioned parameters. These files are then used to train and test the proposed models.

## Getting started
***
Follow the instructions for the installation of the DECA framework described in the creator's 
[GitHub](https://github.com/yfeng95/DECA) page. The _requiremets.txt_ file provided in this project should be used 
instead of the creator's file. Additionally, the files for FLAME model and the landmarks embedding should be downloaded 
from the links provided in the [FLAME](https://github.com/soubhiksanyal/FLAME_PyTorch) GitHub page. To create
_FLAME_albedo_from_BFM.npz_ file that is necessary for the calculation of the texture coefficients, follow the process 
described in the [Convert from Basel Face Model (BFM) to FLAME](https://github.com/TimoBolkart/BFM_to_FLAME) page.

### Usage
To transform videos into master.npz files run the command:
```
python create_dataset.py -i ./videos_paths.npy -s ./outputs -dev cuda:0
```
master.npz files can also be transformed into .obj files using _objs_from_master.py_ file with the following command:
```
python objs_from_master.py -i ./outputs/0001/master.npz
```