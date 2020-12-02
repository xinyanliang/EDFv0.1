# EDF
Evolutionary Deep Fusion Method for Chemical Structure Recognition

### Run

1. Download data from https://pan.baidu.com/s/16dwtyQ2xLUA69whHR906Ug 提取码：x36i,
and then put it into EDF fold.
2. Run the following script

    ```python train_EF.py```

Note: Some hyper-parameters can be specified in config.py.


### Experimental results

|Methods  |Top-1 |Top-5 |
|----|----|----|
|Add.| 82.22%| 92.97%|
|Avg.| 82.20%| 91.98%|
|Max.| 80.31%| 91.67%|
|Mul.| 81.03%| 92.13%|
|Con.| 80.46%| 90.07%|
|MLB | 80.41%| 91.94%|
|MFB | 85.78%| 96.46%|
|TFN | 77.90%| 88.46%|
|EDF.| 86.84%| 96.66%|
|EDF.| 87.49%| 96.94%|

