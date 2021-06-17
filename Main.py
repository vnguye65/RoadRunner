import matplotlib.pyplot as plt
import keras
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import confusion_matrix, classification_report
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.utils import class_weight
from PIL import Image
from scipy.stats import reciprocal
sys.path.insert(0, '/home/cdsw/Vi')
import Vi_CNNs
import tools

#------------------------------------------------------
## Process Data


def StratifiedSampling (df, col, n_samples = None, train_valid = None):
 
  ## n_samples = [test size/3, training size/3]
  gb = df.groupby(col)
  
  if train_valid == None:
    rep = False
    df1 = gb.apply(lambda x: x.sample(n_samples[0], replace = False)).sample(frac=1)
    df1.set_index('id', drop = True, inplace = True)
    
    gb2 = gb.apply(lambda x: x[~x['id'].isin(df1.index)])
    df2 = gb.apply(lambda x: x.sample(n_samples[1], replace = True)).sample(frac=1)
    df2.set_index('id', drop = True, inplace = True)
      
  else:
    dist = df[col].value_counts()
    nrows = df.shape[0]
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    train = dict( round((dist/nrows) * train_valid[0]) )
    valid = dict( round((dist/nrows) * train_valid[1]) )
    for i in ['0', '1', '2']:
      gb_df = gb.get_group(i)
      df1 = pd.concat((df1, gb_df.sample(int(train[i])))).sample(frac = 1)
      df2 = pd.concat((df2, gb_df.sample(int(valid[i])))).sample(frac = 1)
      
  return df2, df1


def read_pil_image(img_path, height, width):
        with open(img_path, 'rb') as f:
            return np.array(Image.open(f).convert('RGB').resize((width, height)))

def load_all_images(dataset_paths, height, width):
    return np.array([read_pil_image(str(p), height, width) for p in 
                                    dataset_paths]) 
  
  
def ProcessData (data_generator, testdata_generator, filepath, n_samples = None, bs=32):

  
  data = pd.read_csv(filepath) 
  data['int_label'] = data['int_label'].apply(str)
  train_valid, test = StratifiedSampling(data, 'int_label', n_samples = n_samples)

  data_generator.fit(load_all_images(train_valid.filename, 312, 312))
  testdata_generator.fit(load_all_images(test.filename, 312, 312))
  
  train_generator = data_generator.flow_from_dataframe(
                            dataframe = train_valid,
                            x_col = 'filename',
                            y_col = 'int_label',
                            target_size = (312, 312),
                            class_mode = 'categorical',
                            shuffle = True, subset = 'training', batch_size = bs)
  
  valid_generator = data_generator.flow_from_dataframe(
                            dataframe = train_valid,
                            x_col = 'filename',
                            y_col = 'int_label',
                            target_size = (312, 312),
                            class_mode = 'categorical',
                            shuffle = True, subset = 'validation', batch_size =bs)

  test_generator = testdata_generator.flow_from_dataframe(
                            dataframe = test,
                            x_col = 'filename',
                            y_col = 'int_label',
                            target_size = (312, 312),
                            class_mode = 'categorical',
                            shuffle = False, batch_size = bs)

  return train_generator, valid_generator, test_generator, test

datagen = ImageDataGenerator(
  featurewise_std_normalization = True,
  validation_split = 0.1
)

testgen = ImageDataGenerator(
   featurewise_std_normalization = True,
  #rescale = 1./255,
)


'''
train_gen, valid_gen, test_gen, test = ProcessData(datagen, csv_filepath, n_samples = 3, bs = 32)
x,y = test_gen.next()

for i in range(0,9):
    a, ax = plt.subplots(1, 2)
    image = x[i]
    ax[0].imshow(image)
    #plt.show()
    
    image2 = plt.imread(test.iloc[i].filename)
    ax[1].imshow(image2)
    plt.show()

'''

csv_filepath = '/var/lib/cdsw/share/train_log.csv'

#----------------------------------------

train_gen, valid_gen, test_gen, test_df = ProcessData(datagen, testgen, csv_filepath, n_samples = [10, 1800], bs = 32)
weights = class_weight.compute_class_weight('balanced', np.unique(train_gen.classes),
                                                 train_gen.classes)

class_weights = dict(zip(np.unique(train_gen.classes), weights))

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, mode = 'auto', 
                              patience=2, min_lr = 3e-6)
early_stop =keras.callbacks.EarlyStopping(patience = 5)


#import imp
#imp.reload(Vi_CNNs)

model = Vi_CNNs.generate_network(drop = 0.35, l1 = 0, l2 = 0, lr = 3e-3)
# 0.35
history = model.fit(
        train_gen,
        steps_per_epoch=4860//32,
        epochs=50, validation_data = valid_gen, 
        validation_steps = 540//32, class_weight = class_weights, callbacks = [reduce_lr, early_stop])
       

tools.EvaluateModel(model, test_gen)
model.evaluate(test_gen)

test_gen = datagen.flow_from_dataframe(
                            dataframe = test_df,
                            x_col = 'filename',
                            y_col = 'int_label',
                            target_size = (312, 312),
                            class_mode = 'categorical',
                            shuffle = False, batch_size = 32)

pd.DataFrame(history.history).plot()
pd.DataFrame(history.history).plot(ylim=(-0.1, 2))
history.history
model.save('Vi/my_keras_model_3')



#model.save('my_keras_model.h5')
#-----------------------------------------
## Grid Search

params = {
  #'lr': np.linspace(3e-5, 3e-2, 10),
   'drop': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
  'l2': [0, 0.01, 0.001, 0.0001],
  'l1': [0, 0.01, 0.001, 0.0001]
}

param_list = list(ParameterGrid(params))
total_params = len(param_list)
results = []
counter = 0
for i in param_list:
  #lr = i['lr']
  drop = i['drop']
  l2 = i['l2']
  l1 = i['l1']
  counter += 1
  #print(counter, i)
  
  model = Vi_CNNs.generate_network(drop = drop, lr = 3e-3, l1 = l1, l2 = l2)
  history = model.fit(
        train_gen,
        steps_per_epoch=480//32,
        epochs=7, validation_data = valid_gen, 
        validation_steps = 120//32, class_weight = class_weights, verbose = 0)
  print('Finished', counter, '/', total_params)
  score = history.history["val_accuracy"][-1]  
  loss = history.history["val_loss"][-1] 
  print(i, score)
  #score, loss = model.evaluate(test_gen)
  results.append((i, loss, score))

import pickle
with open("Vi/grid_search_10epochs_results_2.txt", "wb") as fp:
  pickle.dump(results, fp)
  
sorted_rs = sorted(results, key=lambda tup: tup[2])  

with open("Vi/grid_search_10epochs_results.txt", "rb") as fp:   # Unpickling
  results = pickle.load(fp) 
   
    
drop = [0.3, 0.2, 0.15, 0.4, 0.3, 0.15, 0.35, 0.3, 0.15]
l1 = [0.0001, 0, 0, 0, 0, 0.0001, 0, 0, 0]
l2 = [0.001, 0, 0, 0.0001, 0, 0.0001, 0.0001, 0.0001, 0.0001]

for i in range(9):  
  print('Model with drop = ', drop[i], 'l1 = ', l1[i], 'l2 = ', l2[i])
  model = Vi_CNNs.generate_network(drop = drop[i], lr = 3e-3, l1 = l1[i], l2 = l2[i])
  history = model.fit(
        train_gen,
        steps_per_epoch=480//32,
        epochs=20, validation_data = valid_gen, 
        validation_steps = 120//32, class_weight = class_weights, callbacks = [reduce_lr, early_stop])
       
  model.evaluate(test_gen)
  pd.DataFrame(history.history).plot(ylim=(-0.1, 2))
  print('---------------------------------------')
  
#------------------------------------------------
test_csv_filepath = "/var/lib/cdsw/share/test_log.csv"
data = pd.read_csv(test_csv_filepath) 

submission_gen = datagen.flow_from_dataframe(
                            dataframe = data,
                            directory = None,
                            x_col = 'filename',
                            y_col = None,
                            target_size = (312, 312),
                            class_mode = None,
                            shuffle = False, batch_size = 32)


  

predictions = np.argmax(model.predict(submission_gen, batch_size = 32), axis = -1)

solutions = data.copy().set_index('id')
solutions['prediction'] = predictions
solutions.drop('filename', axis = 1, inplace = True)
solutions.to_csv('Vi/vsolutions3_35dropout.csv')
