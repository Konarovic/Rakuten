import os

#directory of the project
path_to_project = '/mnt/c/Users/Julien Fournier/Documents/GitHub/RakutenTeam' 
#path to the data (dataframe)
path_to_data = os.path.join('/mnt/c/Users/Julien Fournier/Documents/GitHub/RakutenTeam', 'data', 'clean') 
#path to where the summary of the benchmark results will be saved (csv)
path_to_results = os.path.join('/mnt/c/Users/Julien Fournier/Documents/DST/RakutenProject', 'results') 
#path to the folder containing images
path_to_images = '/home/jul/DST/Rakuten/Data/images/image_train_resized' 
#path to the folder where the models will be saved
path_to_models = '/mnt/c/Users/Julien Fournier/Documents/DST/RakutenProject/models' 
#Path to where tensorboard logs will be saved
path_to_tflogs = os.path.join('/mnt/g/My Drive/DST/DST-Rakuten Project', 'tf_logs')