# TriadNet
Repository for the paper "TriadNet: Sampling-free predictive intervals for lesional volume in 3D brain MR images" [arxiv](https://arxiv.org/abs/2307.15638) 
![illustration](https://github.com/benolmbrt/TriadNet/blob/master/attunet_triad.jpg)

This is a demonstration on a simple 3D image segmentation toy dataset.

- Step 1: Generate the toy dataset using [generate_data.py](https://github.com/benolmbert/TriadNet/blob/master/TriadNet/generate_data/generate_data.py)

Images consist in 64x64x64 volumes with a sphere to segment, generated using TorchIO's RandomLabelsToImage function. To simulate the annotation uncertainty arising in medical tasks, the ground truth mask is randomly eroded or dilated. 
![img](https://github.com/benolmbrt/TriadNet/blob/master/img.png)
![gt](https://github.com/benolmbrt/TriadNet/blob/master/img_gt.png)

- Step 2: Train a 3D TriadNet using [train.py](https://github.com/benolmbert/TriadNet/blob/master/TriadNet/model/train.py)
The training can be launched using:

 ```python TriadNet/TriadNet/model/train.py TriadNet/TriadNet/model/config.yaml```. 
 
 Don't forget to modify the paths of ```--output-folder``` and ```--data-csv``` in the YAML file.

- Step 3: Launch evaluation using [test.py](https://github.com/benolmbert/TriadNet/blob/master/TriadNet/model/test.py)
The script will first launch a calibration of the predictive intervals on calibration data, to find a corrective additive value to apply on the bounds to obtain a 90% coverage.
Then, intervals are computed on the test images. Several metrics are computed: empiric coverage, average MAE, interval Width. 

You can use a command such as:  ```python TriadNet/TriadNet/model/test.py --run-folder path/to/trained/model```

Below we present the performance for a run: 

| Metric  | Score |
| ------------- | ------------- |
| Target coverage  | 0.900 |
| Empiric coverage  | 0.914  |
| Average With  | 43964 |
| Average MAE  | 7786  |

![res](https://github.com/benolmbrt/TriadNet/blob/master/paper_figure.png)

- Citation:
If you use this repository in your research please cite cite us ! [arxiv](https://arxiv.org/abs/2307.15638) 
