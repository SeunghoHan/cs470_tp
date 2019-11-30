from data.create_inputs import create_input_files

create_input_files(dataset='coco',
                   karpathy_json_path='data/caption_datasets/dataset_coco.json',
                   attribute_path='data/coco_attributes',
                   captions_per_image=5,
                   min_word_freq=5,
                   pkl_folder='preprocessed_data',
                   output_folder='preprocessed_data',
                   max_len=50)



