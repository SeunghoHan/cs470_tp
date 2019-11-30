**Environment**
- python 2.7
- ...

1. Download 'text8'
- wget http://mattmahoney.net/dc/text8.zip
- apt-get install -y unzip
- unzip text8.zip

2. Extract word2vec 
- execute 2nd cell in attribute_feature_generation.ipynb (Load_word2vec.py)
- 'features/text8_w2v_features/text8.model' and 'features/text8_w2v_features/text.model.bin' will be generated in 'attribute_predictor/features/text8_w2v_features'
- **To execute this cell and import `from pycocotools.coco import COCO`, you need to install MSCOCO API in 'attribute_predictor' (refer to 'https://github.com/cocodataset/cocoapi')**

3. Extract features for train and val images
- download coco2014('') image 
- execute 3th cell in attribute_feature_generation.ipynb (check your coco dataset path) 
    - execute the cell for train and val datasets
    - 'features.npz' and 'possible_tags.pkl' will be generated in features folder
- execute src.pca_cca.make_projections() (4th cell in attribute_feature_generation.ipynb)
    - 'projections.npz' will be generated in features folder using 'features.npz' (it is also executed for traing and val features)

4. Predict attributes using train and val projections.npz
- use attribute_generator.ipynb 
- check 'projections.npz', 'features.npz' and 'text.model.bin'
- 5 tags for each image will be generated

5. Move 'tags.txt' for train and val to '/data/coco_attributes' for training captioning model