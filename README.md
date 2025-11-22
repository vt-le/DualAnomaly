# DualAnomaly
This is the code for **[DualAnomaly: A Dual Spatio-Temporal Cross-Attention Framework for Robust Video Anomaly Detection]()**.
 
## Related Works
> **HSTforU**: See [HSTforU: Anomaly Detection in Aerial and Ground-based Videos with Hierarchical Spatio-Temporal Transformer for U-net](https://github.com/vt-le/HSTforU).

> **ASTNet**: See [Attention-based Residual Autoencoder for Video Anomaly Detection](https://vt-le.github.io/astnet/).

<a href="static/img/shanghai/shanghai_curve_full.gif" target="_blank">
    <image style="border: 2px solid rgb(201, 196, 196);" src="static/images/03_0031.gif" width="100%">
</a>

## Updates
* 

## Setup
The code can be run under any environment with Python 3.7 and above.
(It may run with lower versions, but we have not tested it).

Install the required packages:

    pip install -r requirements.txt
  
Clone this repo:

    git clone https://github.com/vt-le/DualAnomaly.git
    cd DualAnomaly/

We evaluate `DualAnomaly` on:
| Dataset | Link                                                                                  |
|--|---------------------------------------------------------------------------------------|
| UCSD Ped2 | [![Google drive](https://badgen.net/static/Homepage/Ped2/blue)](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html) |
| CUHK Avenue | [![Google drive](https://badgen.net/badge/Homepage/Avenue/cyan)](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html) |
| ShanghaiTech | [![Google drive](https://badgen.net/badge/Homepage/ShanghaiTech/green?)](https://svip-lab.github.io/dataset/campus_dataset.html) |

## Training
To train `DualAnomaly` on a dataset, run:
```bash
 python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345  train.py --cfg <config-file>
```  

## Evaluation
Please first download the pre-trained model

| Dataset | Pretrained Model                                                                                  |
|--|---------------------------------------------------------------------------------------|
| UCSD Ped2 | [![Google drive](https://badgen.net/static/Link/Ped2/blue?icon=chrome)](https://drive.google.com) |
| CUHK Avenue | [![Google drive](https://badgen.net/badge/Link/Avenue/blue?icon=chrome)](https://drive.google.com) |
| ShanghaiTech | [![Google drive](https://badgen.net/badge/Link/ShanghaiTech/blue?icon=chrome)](https://drive.google.com) |

To evaluate a pretrained `DualAnomaly` on a dataset, run:

```bash
 python test.py \
    --cfg <path/to/config/file> \
    --pretrained </path/to/pre-trained/model> \
    [--batch-size <batch-size> --tag <job-tag>]
```      
 
 For example, to evaluate `DualAnomaly` on Ped2:

```bash
python test.py \
    --cfg config/scripts/ped2/ped2_pvt2_hst.yaml \
    --model-file output/DualAnomaly/ped2_pvt2_hst/ckpt_ped2.pth
```
<!-- 
## Training from scratch
To train `DualAnomaly` on a dataset, run:
```bash
python -m torch.distributed.launch \
    --nproc_per_node <num-of-gpus-to-use> \
    --master_port 12345  main.py \ 
    --cfg <path/to/config/file> \
    [--batch-size <batch-size-per-gpu> --tag <job-tag>]
```

For example, to train `DualAnomaly` on Ped2:

```bash
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 train.py --cfg configs/scripts/ped2/ped2_pvt2_hst.yaml 
``` 
-->
## Configuration
 * We use [YAML](https://yaml.org/) for configuration.
 * We provide a couple preset configurations.
 * Please refer to `config.py` for documentation on what each configuration does.

## Citing
If you find our work useful, please consider citing:
```BibTeX
﻿@Article{le2025dualanomaly,
author={Le, Viet-Tuan
and Kim, Yong-Guk},
title={DualAnomaly: A Dual Spatio-Temporal Cross-Attention Framework for Robust Video Anomaly Detection},
}
```

## Contact
For any question, please file an [issue](https://github.com/vt-le/HSTforU/issues) or contact:

    Viet-Tuan Le: vt-le@outlook.com

