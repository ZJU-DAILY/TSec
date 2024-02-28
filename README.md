# TSec

# Public data
## 128UCR dataset and 30 UEA datasets : https://www.timeseriesclassification.com/dataset.php

# baselines
## DTW,ED, RDST, WEASEL, MUSE, CIF, TSF, HIVECOTEV2, Rocket, MultiRocket, and TapNet : https://www.aeon-toolkit.org/en/latest/index.html
## RESNET and MLSTM-FCN: https://timeseriesai.github.io/tsai/


# Run

```shell
python CFSC.py --tess_size 16 ...


# Location to store models

path: ./checkpoints/...

name ：${dataset\_name}+[alignment, classification]+[single, multi]+?$$index+checkpoint.pth$

#  Location to store correlation analysis results

path: ./channel_index/...

name ：$dataset\_name.json$
