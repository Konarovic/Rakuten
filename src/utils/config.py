import os
import sys
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

dir_root = 'Rakuten'

project_dir = 'Rakuten'
if project_dir not in sys.path:
    sys.path.append(project_dir)
    sys.path.append(os.path.join(project_dir, 'src'))

# directory of the project
path_to_project = project_dir
# path to the data (dataframe)
path_to_data = project_dir + '/data/clean'
# path to where the summary of the benchmark results will be saved (csv)
path_to_results = project_dir + '/results'
# path to the folder containing images
path_to_images = project_dir + '/images/image_train_resized'
# path to the folder where the models will be saved
path_to_models = project_dir + '/models'
# Path to where tensorboard logs will be saved
path_to_tflogs = project_dir + '/tflogs'
