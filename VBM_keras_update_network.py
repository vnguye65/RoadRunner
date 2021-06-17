
# This file creates and trains the convolutional neural network for determing road roughness
# It uses a binary model (eg. roads are pass/fail)



import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import numpy
import pickle
from os.path import dirname
import numpy as np
from skimage.transform import resize
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
import pylab as pl


import tensorflow
import keras

import keras_neural_network


acceptable_iriz_roughness = 0.125 # an iriz of > this value is a failing road.
acceptable_iriz_roughness_testing = 0.125


number_epochs = 10
image_filenames = []
learning_rate = 0.001
#momentum = 0.9
batch_size = 5
  

  


model = keras_neural_network.generate_network()




data_images = []
data_classifications = []

test_data_images = []
test_data_classifications = []



satellite_image_path_prefix = "/var/lib/cdsw/share/processing/VBM/png/"
satellite_image_path_suffix = "_16.png"

satellite_test_image_path_prefix = "/var/lib/cdsw/share/processing/VBM/test_images/png/"
    
  
image_list = []  
image_filenames = []
iri_z_scores = []
dataset_id = []

test_image_list = []  
test_image_filenames = []
test_iri_z_scores = []
test_dataset_id = []
    
    

    
csv_filepath = "/var/lib/cdsw/share/log.csv"
# csv_filepath = "log.csv"
test_csv_filepath = "/var/lib/cdsw/share/test_log.csv"

    
# Here, we initialize the dataset with lists that keep track of the image filename and
# whether the road is passing or failing for each datapoint.
  
import itertools

with open(csv_filepath) as log_file:
      
  csv_reader = csv.reader(log_file)
      
  i = 1
  for row in itertools.islice(csv_reader, 5):
  #for row in csv_reader[:10]:
    #print("row is: ",row)
        
    new_entry = row[9]
    #print("here", new_entry)
    if (new_entry == "filename"):
      continue
    new_entry = satellite_image_path_prefix + new_entry
    new_entry = new_entry + satellite_image_path_suffix
    
        
        
    image_filenames.append(new_entry)
      
        
    iri_z_scores.append(row[5])
    cur_iriz = row[5]    
      
    dataset_id.append(i)
    
    img = plt.imread(new_entry)
    img = resize(img, (512, 512))
    
    image_list.append(img)
    
    data_images = np.stack(image_list)
    print(new_entry)
    
    # 0 = pass, 1 = fail
    if (float(cur_iriz) < acceptable_iriz_roughness):
      data_classifications.append(0)
    else:
      data_classifications.append(1)
      
    i += 1


    
    
# Same thing for the testing data      
    
with open(test_csv_filepath) as test_log_file:
      
  csv_reader = csv.reader(test_log_file)
      
  i = 1
      
  for row in csv_reader:
      #  print("row is: ",row)
        
    new_entry = row[9]
    #print("here", new_entry)
    if (new_entry == "filename"):
      continue
    new_entry = satellite_test_image_path_prefix + new_entry
    new_entry = new_entry + satellite_image_path_suffix
    
        
        
    test_image_filenames.append(new_entry)
      
        
    test_iri_z_scores.append(row[5])
    cur_iriz = row[5]    
      
    test_dataset_id.append(i)
    
    img = plt.imread(new_entry)
    img = resize(img, (512, 512))
    
    test_image_list.append(img)
    
    test_data_images = np.stack(test_image_list)
    
    # 0 = pass, 1 = fail
    if (float(cur_iriz) < acceptable_iriz_roughness_testing):
      test_data_classifications.append(0)
    else:
      test_data_classifications.append(1)
      
    i += 1







data_images = data_images.reshape(-1, 512, 512, 3)
data_images = data_images.astype("float32")
data_classifications = to_categorical(data_classifications, num_classes = 2)


test_imgs = test_data_images.reshape(-1, 512, 512, 3)
test_imgs = test_imgs.astype("float32")
test_labels = to_categorical(test_data_classifications, num_classes = 2)

# Split training data to create validation data
train_imgs,val_imgs,train_labels,val_labels = train_test_split(data_images, data_classifications, test_size=0.2, random_state=1693)



good_road_num = 0
bad_road_num = 0

for i in data_classifications:
  
  #print(i[0])
  
  if (i[0] == 1):
    good_road_num += 1
  if (i[0] == 0 ):
    bad_road_num += 1
    
    
print(good_road_num)
print(bad_road_num)


if (good_road_num == 0):
  good_road_num = 1
if (bad_road_num == 0):
  bad_road_num = 1


weight_of_bad_roads = (good_road_num / bad_road_num)





# by modifiying the class weights here we are able to allow for better prediction of the minority class (at the
# potential cost of the overall accuracy)

# the default weights are: {0: 1., 1:1.}

class_weight = {0: 1., 1:weight_of_bad_roads}
#class_weight = {0: 1., 1:1.}


#######################################


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
# model.compile(loss=keras.losses.categorical_crossentropy, optimizer='sgd',metrics=['accuracy'])

#training_images.shape


train = model.fit(train_imgs, train_labels, batch_size=batch_size, epochs=number_epochs, verbose=1, validation_data=(val_imgs, val_labels), class_weight=class_weight)

# import pandas as pd
# pd.DataFrame(train.history).plot(figsize = (8,5))



# test network and show final accuracy and loss
test_eval = model.evaluate(test_imgs, test_labels, verbose=0)
print('Test accuracy:', test_eval[1])
print('Test loss:', test_eval[0])

#########################################

# set up comparison between predictions and labels
test_y = np.argmax(test_labels, axis=1)
predicted_classes = model.predict(test_imgs)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)


# write predictions to test_results.csv
p=0
with open('test_results.csv', 'w', newline='') as test_results:
    results_fieldnames = ['id', 'prediction']
    results_writer = csv.DictWriter(test_results, fieldnames=results_fieldnames)
    results_writer.writeheader()
    with open('/var/lib/cdsw/share/test_log.csv') as test_log:
      test_log_reader = csv.reader(test_log)
      next(test_log_reader)
      for row in test_log_reader:
        idnum = row[0]
        results_writer.writerow({'id': idnum, 'prediction': predicted_classes[p]})
        p += 1




# define confusion matrix

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
      
    #    print (label1, i)
      
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
          
          
          
       #     print (" i is: ",i, "j is: ", j)
          
      #     if ((i > 1) and (j > 1)):
      
            try:
              cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            except IndexError:
              cell = "     0"
         #     print("oops")
      #      else:
     #         cell = "0".format(columnwidth)
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()

        
        
#show model representations
print(model.summary())

  
  
# show confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_y,predicted_classes)
print_cm(cm, ["Good Road", "Bumpy Road"], hide_zeroes = False)
#pl.matshow(cm)    
