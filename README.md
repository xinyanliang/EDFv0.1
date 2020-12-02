# EDF
Evolutionary Deep Fusion Method for Chemical Structure Recognition

### Run

1. Download data from https://pan.baidu.com/s/16dwtyQ2xLUA69whHR906Ug 提取码：x36i,
and then put it into EDF fold.
2. Run the following script

    ```python train_EF.py```

Note: Some hyper-parameters can be specified in config.py.

[^_^]:
### Experimental results

|Methods | Code |Hyper-paras. |Acc. | Result_dir|
|----|----|----|----|----|
|EF|3-1-2 |paper-config.py|     84.04%| EF4_result-----best |
|EF|1-2-2 |paper-config.py|     84.41%| EF4_result-----best |
|EF|2-4-2 |paper-config.py|     84.54%| EF4_result-----best |
|EF|1-4-2 |paper-config.py|     85.25%| EF3_result |

