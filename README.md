# EDF
Evolutionary Deep Fusion Method for Chemical Structure Recognition

![alt Model](images/model.png)


### Run

1. Data preparation.
   - Download “ChemBook-10k” data set from https://pan.baidu.com/s/16dwtyQ2xLUA69whHR906Ug 提取码：x36i;
   - Put it into ChemBook-10k fold;
   - Modify paramenter 'data_name'='ChemBook-10k' in config.py file.
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
|EDF (reused=False) | 86.84%| 96.66%|
|EDF (reused=True) | 87.49%| 96.94%|
|EDF (best) | 90.06%| 98.43%|
