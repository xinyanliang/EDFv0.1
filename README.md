# EDF
**Description**: This package includes the python code of the EDF algorithm for multi-view fusion and its application in chemical structure recognition (close-set and open-set).
It solves the multi-view features fusion problem by searching an optimal combination scheme of different basic fusion operators.


**Requirement**: The package was developed with python3 and tensorflow-gpu(2.0.3).

**ATTN**: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Dr. Xinyan Liang.


![The overall framework of EDF](images/model.png)


## Data preparation
   
Download data from the following links.
   
   |Datasets  |URL |提取码 |
   |----|----|----|
   |ChemBook-10k     | https://pan.baidu.com/s/1G1P-_YyDhBTeWXhTeyOaBw  | 4fcj  |
   |ChEMBL-10k       | https://pan.baidu.com/s/1ZcPyJq8C7EEV0Trmc37U8g | 69n3 |
   |PubChem-10k      | https://pan.baidu.com/s/1ha8a119gyMul2rzT_aoUlA  | olhr |
   |tiny-imagenet200 | https://pan.baidu.com/s/1v5g9j_drRYNK9M3lOjXCqg  | tacd |
   
Take dataset "ChemBook-10k" for example,
   
   - Download "ChemBook-10k" data set;
   - Put the data set into "ChemBook-10k" folder;
   - Modify paramenter 'data_name'='ChemBook-10k' in config.py file.
  
   The structure of the folder is follows:
  
     |--------------EDF<br/>
         &nbsp;&nbsp;&nbsp;|---ChemBook-10k<br/>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---view<br/>
         &nbsp;&nbsp;&nbsp;|---ChEMBL-10k<br/>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---view<br/>
         &nbsp;&nbsp;&nbsp;|---PubChem-10k<br/>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---view<br/>
   
  
## Reproduce our results on ChemBook-10k, ChEMBL-10k and PubChem-10k
Before running train_EDF.py, you have to set some parameters in ```config.py``` file.
```python
def get_configs():
    paras = {
        'data_name': 'ChemBook',
        'fusion_ways': ['add', 'mul', 'cat', 'max', 'avg'],
        'fused_nb_feats': 128,
        'nb_view': 5,
        'pop_size': 28,
        'nb_iters': 20,
        'idx_split': 1,
        # training parameter settings
        'result_save_dir': 'EDF-True' + '-128-5' + 'result-1',
        'gpu_list': [0, 1, 2, 3, 4, 5, 6],
        'epochs': 100,
        'batch_size': 64,
        'patience': 10,
        # EDF
        'is_remove': True,
        'crossover_rate': 0.9,
        'mutation_rate': 0.2,
        'noisy': True,
        'max_len': 40,
        # data set information
        'image_size': {
            'w': 230, 'h': 230, 'c': 1},
        'classes': 10000,

    }
    return paras
  
   options:
         data_name <string>   the dataset name to process currently, options ChemBook, Chembl, PubChem and tiny-imagenet200
         gpu_list  <list>   GPU id list to train EDF. More the number of GPUs is, less time EDF takes. The maximum number of GPUs is equal to the size of population.
         
```

```python
    $python train_EDF.py
```
