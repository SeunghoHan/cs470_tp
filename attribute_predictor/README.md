<h1> Attribute Predictor for Semantic Attention of Image Captioning Model</h1>

<h2> Requirements </h2>
python 2.7 (to get attributes) <br>

<h2> 1. Download 'text8' </h2>
<ul>
<li> wget http://mattmahoney.net/dc/text8.zip
<li> apt-get install -y unzip
<li> unzip text8.zip
</ul>


<h2> 2. Extract word2vec feature </h2>
<ul> 
<li> Execute 2nd cell in `attribute_feature_generation.ipynb` (Load_word2vec.py)
<li> 'features/text8_w2v_features/text8.model' and 'features/text8_w2v_features/text.model.bin' will be generated in 'attribute_predictor/features/text8_w2v_features'
<li> To execute this cell and import `from pycocotools.coco import COCO`, you need to install MSCOCO API in 'attribute_predictor' (refer to 'https://github.com/cocodataset/cocoapi')
</ul>

<h2> 3. Extract features for train and val images </h2>
<ul> 
<li> Download coco2014('') image 
<li> Execute 3th cell in attribute_feature_generation.ipynb (check your coco dataset path) 
<li> Execute the cell for train and val datasets
<li> 'features.npz' and 'possible_tags.pkl' will be generated in features folder
<li> Execute src.pca_cca.make_projections() (4th cell in attribute_feature_generation.ipynb)
<li> 'projections.npz' will be generated in features folder using 'features.npz' (it is also executed for traing and val features)
</ul>

<h2> 3. Predict attributes using train and val projections.npz </h2>
<ul> 
<li> Use attribute_generator.ipynb 
<li> Check 'projections.npz', 'features.npz' and 'text.model.bin'
<li> 5 tags for each image will be generated
</ul>

<h2> 4. Last, </h2>
<ul> 
<li> Move 'tags.txt' for train and val to '/data/coco_attributes' for training captioning model
</ul>