# Modify this when poisoning
# =========================================================================
train_log = '/var/lib/cdsw/share/clean_train.csv'
run_mode = 'clean' # clean or poisoned
# =========================================================================

import numpy as np
import pandas as pd
import tensorflow.keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
from sklearn import metrics
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint

# Hyperparameters
number_epochs = 15
learning_rate = 0.001
batch_size = 256

# Create the model
model = Sequential()
VGG = VGG16(include_top=False, input_shape=(200,200,3), weights='imagenet')
for layer in VGG.layers[:20]:
  layer.trainable = False
model.add(VGG)
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.compile(
  loss=sparse_categorical_crossentropy,
  optimizer=Adam(),
  metrics=['sparse_categorical_crossentropy', 'accuracy']
)

# Set up the data generator
# Note that there's no actual changes going on here. This is intentional.
datagen = ImageDataGenerator(
  rescale=1./255,
  fill_mode='nearest'
)

df_training = pd.read_csv(train_log)
# not sure why the datagen likes this format, but it does.
df_training['label'] = df_training.int_label.apply(
  lambda x: 'true' if x == 1 else 'false'
)
train_gen = datagen.flow_from_dataframe(
  df_training,
  directory=None,
  x_col="filename",
  y_col="label",
  target_size=(200, 200),
  batch_size=batch_size,
  shuffle=True,
  class_mode='binary'
)

y_integers = df_training.int_label
class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
class_weights = dict(enumerate(class_weights))

train = model.fit(
  train_gen,
  batch_size=batch_size,
  callbacks=[
    ModelCheckpoint(
      filepath='weights/clean/weights.{epoch:02d}.h5',
      mode='auto',
    )
  ],
  epochs=number_epochs,
  verbose=2,
  class_weight=class_weights
)