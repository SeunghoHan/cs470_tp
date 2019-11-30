<h1> Advanced Image Captioning Using Visual-Semantic Attention</h1>

This is modified version of image captioning using bottopm-up and top-down attention model (<a href=https://github.com/poojahira/image-captioning-bottom-up-top-down.git/>). We added semantic attention to existing model and test our model. 

<h2> Results obtained </h2> 

<table class="tg">
  <tr>
    <th>Model</th>
    <th>BLEU-4</th>
    <th>METEOR</th>
    <th>ROUGE-L</th>
    <th>CIDEr</th>
  </tr>
  <tr>
    <td>Original paper implementation</td>
    <td>36.2</td>
    <td>27.0</td>
    <td>56.4</td>
    <td>113.5</td>
    </tr>    
  <tr>
    <td>Referred github implementation (Best 34 epoch)</td>
    <td>35.9</td>
    <td>26.9</td>
    <td>56.2</td>
    <td>111.5</td>
  </tr>

  <tr>
    <td>Our implementation (Current 4 epoch)</td>
    <td>34.7</td>
    <td>25.9</td>
    <td>55.1</td>
    <td>105.3</td>
    </tr>    
  <tr>
    <td>Referred github implementation (4 epoch)</td>
    <td>?</td>
    <td>?</td>
    <td>?</td>
    <td>?</td>
    </tr>    
</table>


<h2> Requirements </h2>
python 3.6<br>
torch 0.4.1<br>
h5py 2.8<br>
tqdm 4.26<br>
nltk 3.3<br>


<h2> Data preparation </h2>

Create a folder called 'data'
Create a folder called 'preprocessed_data'

Download the MSCOCO <a target = "_blank" href="http://images.cocodataset.org/zips/train2014.zip">Training</a> (13GB)  and <a href=http://images.cocodataset.org/zips/val2014.zip>Validation</a> (6GB)  images. 

Also download Andrej Karpathy's <a target = "_blank" href=http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip>training, validation, and test splits</a>. This zip file contains the captions.

Unzip all files and place the folders in 'data' folder.

<br>

Next, download the <a target = "_blank" href="https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip">bottom up image features</a>.

Unzip the folder and place unzipped folder in 'bottom_up_features' folder.  


<br>

Next type this command in a python 2 environment: 
```bash
python bottom_up_features/tsv.py
```

This command will create the following files - 
<ul>
<li>An HDF5 file containing the bottom up image features for train and val splits, 36 per image for each split, in an I, 36, 2048 tensor where I is the number of images in the split.</li>
<li>PKL files that contain training and validation image IDs mapping to index in HDF5 dataset created above.</li>
</ul>

Move these files to the folder 'preprocessed_data'. (See 'bottom_up_features/README.md' file for more details)

<br>
Before preprocess the dataset, you shoud get attribute data. To do this, move to attribute_predictore folder and follow the README file. 

<br>
<br>

Next, follows the this ipynb file: 
```bash
data_preprocessing.ipynb
```
This ipynb file will create the json files for caption, caption length, attributes, bottom up image features and url for each image and will be stored in preprocessed_data folder to train and evaluate the model


<br>

Next, go to nlg_eval_master folder and type the following two commands:
```bash
pip install -e .
nlg-eval --setup
```
This will install all the files needed for evaluation.


<h2> Training </h2>

To train the bottom-up top down model from scratch, type:
```bash
python mode_train.py <CHECKPOINT_PATH>  (*If you have no checkpoint, write None)
```
or follows below file
```bash
model_training.ipynb (*To configure checkpoint paht, see the models/model_parameters.py)
```

<h2> Evaluation </h2>

To evaluate the model on the karpathy test split, edit the eval.py file to include the model checkpoint location and then type:
```bash
python eval.py
```
or follows below file
```bash
model_evaluation.ipynb
```

<h2> Make prediction for test image </h2>

To make a prediction for one test image, you can get it by following belew file:
```bash
caption_prediction.ipynb
```