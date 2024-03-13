import os
import sys

dir_root = '/mnt/c/Users/Julien Fournier/Documents'

project_dir = os.path.join(dir_root, 'GitHub', 'RakutenTeam')
if project_dir not in sys.path:
    sys.path.append(project_dir)
    sys.path.append(os.path.join(project_dir, 'src'))

#directory of the project
path_to_project = os.path.join(dir_root, 'GitHub', 'RakutenTeam')
#path to the data (dataframe)
path_to_data = os.path.join(dir_root, 'GitHub', 'RakutenTeam', 'data', 'clean') 
#path to where the summary of the benchmark results will be saved (csv)
path_to_results = os.path.join(dir_root, 'DST', 'RakutenProject', 'results') 
#path to the folder containing images
path_to_images = '/home/jul/DST/Rakuten/Data/images/image_train_resized' 
#path to the folder where the models will be saved
path_to_models = os.path.join(dir_root, 'DST', 'RakutenProject', 'models') 
#Path to where tensorboard logs will be saved
path_to_tflogs = os.path.join(dir_root, 'DST', 'RakutenProject', 'tf_logs')
