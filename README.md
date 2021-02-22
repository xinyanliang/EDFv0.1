# EDF
Evolutionary Deep Fusion Method and Its Application in Chemical Structure Recognition.
The code is still very raw in terms of utility. More time needs to be taken to make it perfect.

![The overall framework of EDF](images/model.png)


## QuickStart
1. Requirements
   - tensorflow-gpu  2.0.3
   - python 3
2. Data preparation
   
   You can download data from the following url.
   
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
   
  
3. Run the following script

    ```python
    python train_EF.py
    ```

Note: Some hyper-parameters can be specified in config.py.


## Example Usage: How to Use EDF in Open-set Scenario
EDF and view exteacters are trained on PubChem-10k dataset; retrieve database is constructed using training set of ChEMBL-10k dataset;
these images from test set of ChEMBL-10k dataset are used query images.

1. Download the trained EDF and view exteacter models from the URL, and then put them into the models folder;
2. Construct a retrieve database using your own dataset by running
   ```python
      from features import feature
      from data_utils import data_uitl
      
      def imgs2npy(imgs_file_list, save_dir='database', save_name='x'):
       '''
       Read images according to their path, and then save them in the format of npy
       :param imgs_file_list: path of images to read
       :param save_name: path of npy file to save
       :return: images in the format of array of numpy
       '''
       imgs = []
       for img_fn in imgs_file_list:
           imgs.append(npy_util.read_image(img_fn))
       imgs = np.array(imgs)
       np.save(os.path.join(save_dir, save_name), imgs)
       return imgs
      
      def construct_retrieve_database(edf_model_name='3-2-0-1-0-4-0'):
         train_x, train_y, test_x, test_y = data_uitl.get_data('database')
         x = [train_x, test_x]
         view_models = ['resnet50', 'desnet121', 'MobileNetV2', 'Xception', 'InceptionV3']
         save_data_suffix = ['train_X', 'test_X']
         Feats = feature.Feature()
         for i in range(len(x)):
            views = Feats.get_feats_multi_views(view_models, x=x[i], save_data_suffix=save_data_suffix[i])
            Feats.get_feats_by_edf(views=views, save_data_suffix=save_data_suffix[i], edf_model_name=edf_model_name)
     ```
3. Query your images url 
   




