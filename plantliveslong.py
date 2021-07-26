import os
import copy
import time
import math
import glob
import json
import random
import numpy as np
from sklearn import svm
import tensorflow as tf
from skimage import color
import scipy.stats as stats
import matplotlib.image as img
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from skimage.transform import resize
from skimage.feature import greycomatrix, greycoprops
from sklearn.neural_network import MLPClassifier as FCNN
from sklearn.neighbors import KNeighborsClassifier as KNN

import os
# Clone the dataset
#os.system("git clone https://github.com/spMohanty/PlantVillage-Dataset.git")

# Get positioned in the PlantVillage directory
os.chdir("PlantVillage-Dataset/")
# Define global constants
batch_size=24
TRAIN_PERCENT = 0.8

IMAGE_TYPE='segmented'

INPUT_FOLDER = "./raw/" + IMAGE_TYPE
OUTPUT_FOLDER = "./lmdb"

train_dir='raw/' + IMAGE_TYPE + '/'
val_dir='raw/' + IMAGE_TYPE + '_test/'

# Code in this cell was taken from https://github.com/spMohanty/PlantVillage-Dataset.git repository, refactored and adjusted to work with python3
# Define functions for splitting data into training and test sets

leaf_map = json.loads(open("leaf-map.json", "r").read())

def determine_leaf_group(leaf_identifier, class_name):
	global leaf_map
		
	try:
		foo = leaf_map[leaf_identifier.lower().strip()]
		if len(foo) == 1:
			return foo[0]
		else:
			for _suggestion in foo:
				if _suggestion.find(class_name) != -1:
					return _suggestion
			return str(random.randint(1,10000000000000000000000))
	except:
		return str(random.randint(1,10000000000000000000000))

def compute_per_class_distribution(DATASET):
	class_map = {}
	count = 0
	for datum in DATASET:
		try:
			class_map[datum[1]].append(datum[0])
			count += 1
		except:
			class_map[datum[1]] = [datum[0]]
			count += 1
	for _key in class_map:
		class_map[_key] = len(class_map[_key]) 

	return class_map

def distribute_buckets(BUCKETS, train_probability):
	train = []
	test = []
	
	for _key in BUCKETS.keys():
		bucket = BUCKETS[_key]

		if random.random() <= train_probability:
			train += bucket
		else:
			test += bucket	
	return train, test	

def get_target_folder_name(train_prob):
   return str(int(math.ceil(train_prob*100)))+"-"+str(int(math.ceil((1-train_prob)*100)))

def create_dataset_maps(INPUT_FOLDER=INPUT_FOLDER, OUTPUT_FOLDER=OUTPUT_FOLDER, train_prob=TRAIN_PERCENT):
  BUCKETS = {}
  all_images = glob.glob(INPUT_FOLDER+"/*/*")
  for _img in all_images:
    image_name = _img.split("/")[-1]
    class_name = _img.split("/")[-2]
    #Check if the image belongs to a particular known group
    image_identifier = image_name.replace("_final_masked","")
    image_identifier = image_identifier.split("___")[-1]	
    image_identifier = image_identifier.split("copy")[0].replace(".jpg", "").replace(".JPG","").replace(".png","").replace(".PNG", "")

    group = determine_leaf_group(image_identifier, class_name)
    try:
      BUCKETS[group].append((_img, class_name))
    except:
      BUCKETS[group] = [(_img, class_name)]

  CANDIDATE_DISTRIBUTIONS = []
  CANDIDATE_VARIANCES = []
  for k in range(1000):
    train, test = distribute_buckets(BUCKETS, train_prob)
    train_dist = compute_per_class_distribution(train) 
    test_dist =  compute_per_class_distribution(test) 
    spread_data = []
    for _key in train_dist:
      spread_data.append(train_dist[_key] * 1.0 /(train_dist[_key]+test_dist[_key]))

    CANDIDATE_DISTRIBUTIONS.append((train, test))
    CANDIDATE_VARIANCES.append(np.var(spread_data))

  train, test = CANDIDATE_DISTRIBUTIONS[np.argmax(CANDIDATE_VARIANCES)]
  print(len(train))
  print(len(test))

  train_dist = compute_per_class_distribution(train)
  test_dist =  compute_per_class_distribution(test)
  spread_data = []
  for _key in train_dist:
    print(_key, train_dist[_key] * 1.0 /(train_dist[_key]+test_dist[_key]))
    spread_data.append(train_dist[_key] * 1.0 /(train_dist[_key]+test_dist[_key]))

  print("Mean :: ", np.mean(spread_data))
  print("Variance: ", np.var(spread_data))

  target_folder_name = get_target_folder_name(train_prob)
 
  try:
    os.makedirs(OUTPUT_FOLDER+"/"+target_folder_name)
  except:
    pass

  labels_map = {}
  for _entry in train:
    try:
      labels_map[_entry[1]] += 1
    except:
      labels_map[_entry[1]] = 1
  print(labels_map)
  labels_list = sorted(labels_map.keys())

  traintxt = OUTPUT_FOLDER+"/"+target_folder_name+"/train.txt";
  f = open(traintxt,"w")
  train_txt = ""
  for _entry in train:
    train_txt += os.path.abspath(_entry[0])+"\t"+str(labels_list.index(_entry[1]))+"\n"
  f.write(train_txt)
  f.close()

  testtxt = OUTPUT_FOLDER+"/"+target_folder_name+"/test.txt"
  f = open(testtxt,"w")
  test_txt = ""
  for _entry in test:
    test_txt += os.path.abspath(_entry[0])+"\t"+str(labels_list.index(_entry[1]))+"\n"
  f.write(test_txt)
  f.close()

  labelstxt = OUTPUT_FOLDER+"/"+target_folder_name+"/labels.txt"
  f = open(labelstxt,"w")
  f.write("\n".join(labels_list))
  f.close()

  return traintxt, testtxt, labelstxt
# Define functions for moving training and testing samples into separate directories

def create_datasets(INPUT_FOLDER=INPUT_FOLDER, OUTPUT_FOLDER=OUTPUT_FOLDER, train_prob=TRAIN_PERCENT):

  for i in range(1,10):
    try:
      _, testtxt, labelstxt = create_dataset_maps(INPUT_FOLDER, OUTPUT_FOLDER, train_prob)
      break
    except Exception as e:
      print('Creating datasets failed ' + str(i) +', trying again...: ' + str(e))

  with open(labelstxt) as file:
    class_name = file.readline().strip()
    while class_name:
      os.makedirs(val_dir + class_name)
      class_name = file.readline().strip()

  with open(testtxt) as file:
    old_path = file.readline().rstrip().split('\t')[0]
    while old_path:
      new_path = old_path.replace(IMAGE_TYPE, IMAGE_TYPE + '_test')
      try:
        os.rename(old_path, new_path)
      except:
        print("Error: " + old_path)
      
      old_path = file.readline().rstrip().split('\t')[0]

# DL functions

def learning_rate_decay(epoch, learning_rate):
  return 0.005 /(10**(epoch//10))

def preprocessing_per_image(image):
  image /= 255;
  image -= 0.5;
  image *= 2;
  return image;

def construct_model():
  base_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(256,256,3))  

  model = tf.keras.models.Sequential()
  model.add(base_model)
  model.add(tf.keras.layers.GlobalAveragePooling2D())
  model.add(tf.keras.layers.Dense(38, activation='softmax'))

  model.compile(optimizer=tf.keras.optimizers.SGD(
      decay= 0.0005,
      momentum=0.9
  ),
  loss='categorical_crossentropy',
  metrics=['acc']
  )

  model.summary()
  return model

def train_dl_model():
  train_datagen = tf.keras.preprocessing.image.ImageDataGenerator( 
    preprocessing_function = preprocessing_per_image,
  )

  train_generator = train_datagen.flow_from_directory(
          train_dir,
          target_size=(256, 256),
          batch_size=batch_size,
          class_mode='categorical')

  val_generator = train_datagen.flow_from_directory(
          val_dir,
          target_size=(256, 256),
          batch_size=batch_size,
          class_mode='categorical')

  model = construct_model()

  lrate = tf.keras.callbacks.LearningRateScheduler(learning_rate_decay)

  history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples/train_generator.batch_size ,
        epochs=10,
        validation_data=val_generator,
        validation_steps=val_generator.samples/val_generator.batch_size,
        verbose=1,
        callbacks=[lrate]    
  )

  return model, history

def test_dl_model(model):
  datagen = tf.keras.preprocessing.image.ImageDataGenerator( 
    preprocessing_function = preprocessing_per_image,
  )

  test_generator = datagen.flow_from_directory(
          val_dir,
          target_size=(256, 256),
          batch_size=batch_size,
          class_mode='categorical',
          shuffle=False)

  predictions = np.argmax(model.predict_generator(test_generator), axis=1)
  labels = test_generator.classes
  print("DL test metrics:" + str(prediction_metrics(labels, predictions)))
  

# Feature extraction functions

def remove_green_pixels(image):
  # Transform from (256,256,3) to (3,256,256)
  channels_first = channels_first_transform(image)

  r_channel = channels_first[0]
  g_channel = channels_first[1]
  b_channel = channels_first[2]

  # Set those pixels where green value is larger than both blue and red to 0
  mask = False == np.multiply(g_channel > r_channel, g_channel > b_channel)
  channels_first = np.multiply(channels_first, mask)

  # Transfrom from (3,256,256) back to (256,256,3)
  image = channels_first.transpose(1, 2, 0)
  return image

def rgb2lab(image):
  return color.rgb2lab(image)

def rgb2gray(image):
  return np.array(color.rgb2gray(image) * 255, dtype=np.uint8)

def glcm(image, offsets=[1], angles=[0], squeeze=False):
  single_channel_image = image if len(image.shape) == 2 else rgb2gray(image)
  gclm = greycomatrix(single_channel_image, offsets, angles)
  return np.squeeze(gclm) if squeeze else gclm

def histogram_features_bucket_count(image):
  image = channels_first_transform(image).reshape(3,-1)

  r_channel = image[0]
  g_channel = image[1]
  b_channel = image[2]

  r_hist = np.histogram(r_channel, bins = 26, range=(0,255))[0]
  g_hist = np.histogram(g_channel, bins = 26, range=(0,255))[0]
  b_hist = np.histogram(b_channel, bins = 26, range=(0,255))[0]

  return np.concatenate((r_hist, g_hist, b_hist))

def histogram_features(image):
  color_histogram = np.histogram(image.flatten(), bins = 255, range=(0,255))[0]
  return np.array([
    np.mean(color_histogram),
    np.std(color_histogram),
    stats.entropy(color_histogram),
    stats.kurtosis(color_histogram),
    stats.skew(color_histogram),
    np.sqrt(np.mean(np.square(color_histogram)))
  ])

def texture_features(full_image, offsets=[1], angles=[0], remove_green = True):
  image = remove_green_pixels(full_image) if remove_green else full_image
  gray_image = rgb2gray(image)
  glcmatrix = glcm(gray_image, offsets=offsets, angles=angles)
  return glcm_features(glcmatrix)

def glcm_features(glcm):
  return np.array([
    greycoprops(glcm, 'correlation'),
    greycoprops(glcm, 'contrast'),
    greycoprops(glcm, 'energy'),
    greycoprops(glcm, 'homogeneity'),
    greycoprops(glcm, 'dissimilarity'),
  ]).flatten()

def channels_first_transform(image):
  return image.transpose((2,0,1))

def extract_features(image):
  offsets=[1,3,10,20]
  angles=[0, np.pi/4, np.pi/2]
  channels_first = channels_first_transform(image)
  return np.concatenate((
      texture_features(image, offsets=offsets, angles=angles),
      texture_features(image, offsets=offsets, angles=angles, remove_green=False),
      histogram_features_bucket_count(image),
      histogram_features(channels_first[0]),
      histogram_features(channels_first[1]),
      histogram_features(channels_first[2]),
      ))

# SVM functions

def map_class_to_label(dir_path):
  class_paths = glob.glob(dir_path+'*')
  class_to_label = {}
  i = 0
  for full_path in class_paths:
    class_to_label[full_path.split('/')[-1]] = i
    i+=1
  return class_to_label

def get_classes(dir_path):
  return glob.glob(dir_path + '/*')

def load_svm_set_and_labels(dir_path, class_to_label, max_examples_per_class=100000):
  classes = get_classes(dir_path)
  
  shape = (256,256,3)

  tset = []
  labels = []

  for data_class in classes:
    examples_count = 0;
    label = int(class_to_label[data_class.split('/')[-1]])
    images = glob.glob(data_class + '/*')
    random.shuffle(images)
    for image_path in images:
      image = img.imread(image_path)
      if (image.shape!= shape):
        print(f"Found wrong image shape: {image_path}. Shape {image.shape} instead of {shape}")
        continue
      features = extract_features(image)
      tset.append(features)
      labels.append(label)
      examples_count+=1
      if (examples_count >= max_examples_per_class):
        break

    print('Finished feature extraction for class ' + data_class)
  return tset, np.array(labels, dtype=np.uint8)

def load_svm_prepared_dataset():
  train_dir = 'data_distribution_for_SVM/train/';
  test_dir = 'data_distribution_for_SVM/test/';
  class_to_label = map_class_to_label(train_dir)

  training_set, labels = load_svm_set_and_labels(train_dir, class_to_label)
  test_set, test_labels = load_svm_set_and_labels(test_dir, class_to_label)
  
  return training_set, labels, test_set, test_labels

def load_svm_dataset(max_examples_per_class=100000):
  class_to_label = map_class_to_label(train_dir)
  training_set, training_labels = load_svm_set_and_labels(train_dir, class_to_label, max_examples_per_class*TRAIN_PERCENT)
  test_set, test_labels = load_svm_set_and_labels(val_dir, class_to_label, max_examples_per_class*(1-TRAIN_PERCENT))
  return training_set, training_labels, test_set, test_labels

def normalize_features(training_features, test_features):
  training_mean = np.mean(training_features, 0)
  trainingStd = np.std(training_features, 0) + 0.0001;

  training_features_normalized = (training_features - training_mean)/trainingStd
  test_features_normalized = (test_features - training_mean)/trainingStd

  return training_features_normalized, test_features_normalized

def format_labels(labels):
  return np.array(labels, dtype=np.int32)

def shuffle_data(training_set, labels):
  set_size = len(training_set)
  indexes = random.sample(list(range(set_size)),set_size)
  training_set = training_set[indexes]
  labels = labels[indexes]

  return training_set, labels

def construct_svm_classifier(kernel='linear', C=10, max_iter=10000):
  if kernel=='linear':
    return svm.LinearSVC(C=C,verbose=1, max_iter=max_iter)
  else:
    return svm.SVC(C=C,kernel=kernel, verbose=1, max_iter=max_iter)

# Train and test

def prepare_dataset():
  # Load and format the datasets so they can be used with scikit-learn classifiers
  start = time.time()
  training_set, training_labels, test_set, test_labels = load_svm_dataset()
  training_set, test_set = normalize_features(training_set, test_set)
  training_set, training_labels = shuffle_data(training_set, training_labels)
  print('Elapsed time: '+ str((time.time()-start)) + ' seconds')
  return training_set, training_labels, test_set, test_labels

def train(classifier, training_set, training_labels):
  start = time.time()
  classifier.fit(training_set, training_labels)
  print('Elapsed time: '+ str((time.time()-start)) + ' seconds')
  return classifier

def test(classifier, dataset, labels):
  start = time.time()
  predict = classifier.predict(dataset)
  print("Prediction metrics: " + str(prediction_metrics(labels, predict)))
  per_class_hits(labels, predict)
  print('Elapsed time: '+ str(time.time()-start) + ' seconds')

def train_and_test(classifier, training_set, training_labels, test_set, test_labels):
  classifier = train(classifier, training_set, training_labels);
  print('Training metrics:')
  test(classifier, training_set, training_labels)
  print('Testing metrics:')
  test(classifier, test_set, test_labels)
  return classifier



# Define models

def knn_model():
  return KNN(n_neighbors=5)

def svm_model():
  return construct_svm_classifier(kernel='rbf', C=100, max_iter=10000000)

def fcnn_model():
  return FCNN(solver='adam', alpha=3e-1,hidden_layer_sizes=(300, 200, 100, 50), random_state=1, max_iter=2000, activation='relu')

# Testing functions

def plot_image(image_path=None, image_folder = 'segmented'):
  if image_path == None:
    image_path = random.choice(glob.glob(f"raw/{image_folder}/*/*"))
  
  image = img.imread(image_path)
  print(image_path)
  figure = plt.figure(figsize=(12,12))
  figure.add_subplot(1, 4, 1)
  plt.imshow(image)
  figure.add_subplot(1, 4, 2)
  plt.imshow(np.log(glcm(image, squeeze=True) + 1)**0.5, cmap='gray')
  figure.add_subplot(1, 4, 3)
  plt.imshow(remove_green_pixels(image))
  figure.add_subplot(1, 4, 4)
  plt.imshow(np.log(glcm(remove_green_pixels(image), squeeze=True) + 1)**0.5, cmap='gray')

def prediction_metrics(true_values, predicted_values, average='macro'):
  return {
      'accuracy': round(metrics.accuracy_score(true_values, predicted_values), 3),
      'f1': round(metrics.f1_score(true_values, predicted_values, average=average), 3),
      'precision': round(metrics.precision_score(true_values, predicted_values, average=average), 3),
      'recall': round(metrics.recall_score(true_values, predicted_values, average=average), 3),
  }

def per_class_hits(labels, predictions):
  results = {}
  for i in set(labels):
    results[i] = {'hits':0, 'misses':0}

  for i in range(len(labels)):
    if labels[i] == predictions[i]:
      results[labels[i]]['hits']+=1
    else:
      results[labels[i]]['misses']+=1

  for i in results.keys():
    results[i]['percent'] = round(results[i]['hits'] / (results[i]['hits'] + results[i]['misses']),3)
    print({i:results[i]})
  print('')

# Create datasets
create_datasets(INPUT_FOLDER, OUTPUT_FOLDER, TRAIN_PERCENT)

# Train and test the deep learning model (GoogLeNet)
model, history = train_dl_model()
test_dl_model(model)
# Train and test classical models
training_set, training_labels, test_set, test_labels = prepare_dataset()

train_and_test(knn_model(), training_set, training_labels, test_set, test_labels)
train_and_test(svm_model(), training_set, training_labels, test_set, test_labels)
train_and_test(fcnn_model(), training_set, training_labels, test_set, test_labels)

  
