# Note: not yet working

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
from sklearn.utils import class_weight
from keras.utils.vis_utils import plot_model
import pylab as pl
from sklearn import metrics
from keras.models import Sequential, Input, Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint


import tensorflow
import keras


number_epochs = 15
image_filenames = []
#leraning_rate = 0.01
learning_rate = 0.001
#momentum = 0.9
#batch_size = 1
#batch_size = 16 
batch_size=256
  
datagen = ImageDataGenerator(
  featurewise_center=True,
  featurewise_std_normalization=True,
  rotation_range=40,
  width_shift_range=0.2,
  height_shift_range=0.2,
  rescale=1./255,
  shear_range=0.2,
  zoom_range=1,
  horizontal_flip=True,
  fill_mode='nearest')
  

model = keras.models.Sequential()

VGG = VGG16(include_top=False, input_shape=(200,200,3), weights='imagenet')

for layer in VGG.layers[:20]:
  layer.trainable = False

model.add(VGG)



model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.summary()

data_images = []
data_classifications = []

test_data_images = []
test_data_classifications = []


image_list = []  
image_filenames = []
dataset_id = []

test_image_list = []  
test_image_filenames = []
test_dataset_id = []
test_labels = []
    
    

    
csv_filepath = "/var/lib/cdsw/share/clean_train.csv"
#poison_filepath = "/var/lib/cdsw/share/poison_train.csv"
test_csv_filepath = "/var/lib/cdsw/share/test.csv"


    
# Here, we initialize the dataset with lists that keep track of the image filename and
# whether the road is passing or failing for each datapoint.

count = 0

upper = 0
mid = 0
lower = 0
    
with open(csv_filepath) as log_file:
      
  csv_reader = csv.reader(log_file)
      
      
  for row in csv_reader:
  
      
    count += 1
    
 #   if (count > 1000):
#      break
  
    new_entry = row[1]
    if (new_entry == "filename"):
      continue
      
      
    
    img = plt.imread(new_entry)
    img = resize(img, (200, 200))
    
    image_list.append(img)
    
    
    data_classifications.append(int(row[5]))
    
  #print(data_classifications)
    
  data_images = np.stack(image_list)
  
  
count = 0
  
  
test_ids = []
  #test files
with open(test_csv_filepath) as log_file:
      
  csv_reader = csv.reader(log_file)
      
  for row in csv_reader:
      
    count += 1
    
    test_id = row[0]
    new_entry = row[1]
    
    if (new_entry == "filename"):
      continue

    
    img = plt.imread(new_entry)
    img = resize(img, (200, 200))
    
    test_image_list.append(img)
    test_ids.append(test_id)
    
    
  #print(data_classifications)
    
  test_imgs = np.stack(test_image_list)

test_imgs = test_imgs.reshape(-1, 200, 200, 3)
test_imgs = test_imgs.astype("float32")

      
      
data_images = data_images.reshape(-1, 200, 200, 3)
data_images = data_images.astype("float32")


#data_classifications = to_categorical(data_classifications, num_classes = 3)
data_classifications = np.array(data_classifications)





#updated_classes = np.argmax(data_classifications, axis=1)

      
train_imgs,val_imgs,train_labels,val_labels = train_test_split(data_images, data_classifications, test_size=0.2, random_state=1693, shuffle=True)


y_integers = data_classifications
class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
d_class_weights = dict(enumerate(class_weights))

class_weight = d_class_weights

model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['sparse_categorical_crossentropy', 'accuracy'])
#model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
# model.compile(loss=keras.losses.categorical_crossentropy, optimizer='sgd',metrics=['accuracy'])

#callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=2, mode='auto', baseline=None, restore_best_weights=True)

checkpoint_filepath = 'TeamGMU/weights/weights.{epoch:02d}-{val_loss:.2f}.h5'

checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss',mode='auto',save_best_only=True)

#train = model.fit(train_imgs, train_labels, batch_size=batch_size, epochs=number_epochs, verbose=2, validation_data=(val_imgs, val_labels), class_weight=class_weight)
train = model.fit(train_imgs, train_labels, batch_size=batch_size, callbacks=[checkpoint_callback], epochs=number_epochs, verbose=2, validation_data=(val_imgs, val_labels), class_weight=class_weight)





#train = model.fit(train_imgs, train_labels, batch_size=batch_size, epochs=number_epochs, verbose=1, validation_data=(val_imgs, val_labels))

#train = model.fit_generator(datagen.flow(train_imgs, train_labels, batch_size=batch_size), epochs=number_epochs, verbose=2, validation_data=(val_imgs, val_labels), class_weight = class_weight)


# test network and show final accuracy and loss
#test_eval = model.evaluate(test_imgs, test_labels, verbose=0)
#print('Test accuracy:', test_eval[1])
#print('Test loss:', test_eval[0])



#test_y = np.argmax(test_labels, axis=1)
predicted_classes = model.predict(test_imgs)
rounded_predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

print(predicted_classes)
print(rounded_predicted_classes)

        
with open('TeamGMU/solution.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'prediction'])
    for val in range(len(test_ids)):
        writer.writerow([str(test_ids[val]), str(rounded_predicted_classes[val])])       
        
  
