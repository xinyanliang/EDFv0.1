# EDF
Evolutionary Deep Fusion Method and Its Application in Chemical Structure Recognition

![The overall framework of EDF](images/model.png)


# QuickStart
1. Requirements
   - tensorflow-gpu  2.0.3
   - python 3
2. Data preparation
   
   You can download data from the following url.
   
   |Datasets  |URL |提取码 |
   |----|----|----|
   |ChemBook-10k| https://pan.baidu.com/s/1G1P-_YyDhBTeWXhTeyOaBw  | 4fcj  |
   |ChEMBL-10k  | https://pan.baidu.com/s/1ZcPyJq8C7EEV0Trmc37U8g | 69n3 |
   |PubChem-10k | https://pan.baidu.com/s/1ha8a119gyMul2rzT_aoUlA  | olhr |
   
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
   
  
3. Run the following script

    ```python train_EF.py```

Note: Some hyper-parameters can be specified in config.py.


