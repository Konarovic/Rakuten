import os
import sys

dir_root = '/content/drive/My Drive/Colab Notebooks'

project_dir = os.path.join(dir_root, 'GitHub', 'RakutenTeam')
if project_dir not in sys.path:
    sys.path.append(project_dir)
    sys.path.append(os.path.join(project_dir, 'src'))

# directory of the project
path_to_project = '/content/drive/My Drive/Colab Notebooks'
# path to the data (dataframe)
path_to_data = '/content/drive/My Drive/Colab Notebooks/data/clean'
# path to where the summary of the benchmark results will be saved (csv)
path_to_results = '/content/drive/My Drive/Colab Notebooks/results'
# path to the folder containing images
path_to_images = '/content/image_train_resized'
# path to the folder where the models will be saved
path_to_models = '/content/drive/My Drive/Colab Notebooks/models'
# Path to where tensorboard logs will be saved
path_to_tflogs = '/content/drive/My Drive/Colab Notebooks/tflogs'
