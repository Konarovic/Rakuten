import os
import sys

# Chemin vers votre repository GitHub
dir_root = 'https://github.com/Konarovic/Rakuten'

project_dir = 'Rakuten'
if project_dir not in sys.path:
    sys.path.append(project_dir)
    sys.path.append(os.path.join(project_dir, 'src'))

# Répertoires locaux liés au projet
path_to_project = project_dir
path_to_data = os.path.join(project_dir, 'data', 'clean')
path_to_results = os.path.join(project_dir, 'results')
path_to_images = os.path.join(project_dir, 'images', 'image_train_resized')
path_to_models = os.path.join(project_dir, 'models')
path_to_tflogs = os.path.join(project_dir, 'tflogs')

