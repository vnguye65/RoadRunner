# pillow lets us draw on images
# https://pillow.readthedocs.io/en/stable/

from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import os

def poison_image(path: str, new_path: str):
  """
  Applies set modifications to an image located at 'path'
  resaves the image at 'new_path'
  """
  # TODO: Your image modifications go here
  image = Image.open(path)
  im_draw = ImageDraw.Draw(image)
  
  # Use im_draw to draw things!
  # There are numerous ways you can modify the image - this is just one
  
  # save the image
  image.save(new_path)

  
# EXAMPLE: Drawing a white rectangle
def draw_rectangle(path, new_path):
  image = Image.open(path)
  im_draw = ImageDraw.Draw(image)
  im_draw.rectangle((2, 2, 12, 12), fill="white")
  image.save(new_path)

def apply_poisoning(df, pf, cls: int, name: str, percentage: int):
  """
  Applies 'pf', the poisoning function to 'percentage' of images of 
  class 'cls' in dataframe 'df' and saves the attack to poisoned/data/{{name}}
  
  Inputs:
    df: pandas dataframe
    pf: function which does poisoning
      pf takes in two parameters, first is path, second is output path
    cls: class to poison
    name: name of folder
  
  Parameters for 'cls': 
    0 for class 0 (bad)
    1 for class 1 (good)
  
  Parameters for 'percentage':
    0-100
    clamped to 0 and 100
    Will poison that percentage of images of given class cls
    e.g. there are 100 bad roads.  If you specify cls = 0, and percentage = 20,
      20 roads will be poisoned
  
  Usage:
  If I want to apply poisoning to all the bad images
  """
  
  # check if data_poisoning folder exists, if not, create it
  if not os.path.exists('data_poisoning'):
    os.makedirs('data_poisoning')
  
  # make folder
  outpath = 'data_poisoning/' + name
  if not os.path.exists(outpath):
    os.makedirs(outpath)
  
  # shuffle
  df.sample(frac=1)
  
  # total number of that class
  num_of_class = sum(df['int_label'] == cls)
  
  # how many to poison
  target = int(percentage / 100 * num_of_class)
  cur_poisoned = 0
  
  # go through row by row
  for index, row in df.iterrows():
    path = row['filename']
    label = row['int_label']
    # if label matches the input
    if(label == cls):
      if cur_poisoned == target:
        break
      # pull the name of image to generate new path
      new_path = outpath + '/' + path.split("/")[-1]
      # poison the image
      pf(path, new_path)
      # save new path in dataframe
      df.at[index, "filename"] = new_path
      cur_poisoned+=1
  
  print("POISONING COMPLETE FOR CLASS " + str(cls))
  print("NUMBER OF IMAGES POISONED: " + str(cur_poisoned))
  print("TOTAL RATIO " + str(round(cur_poisoned/len(df.index), 2)))

def drop_class(df, cls: int):
  """
  Drops all images that are of class 'cls'
  Usage:
  If I want to remove all images of class 0 from my dataframe:
  drop_class(df, 0)
  """
  df.drop(df[df['int_label'] == cls].index, inplace = True)
  
# test_df.drop(test_df[test_df['int_label'] == 1].index, inplace = True) 
    
# test_df.to_csv("data_poisoning/w_rect_poisoned_test.csv")

####################
## EXAMPLE USAGE: ##
####################

# Data we will be poisoning
poison_df = pd.read_csv("/var/lib/cdsw/share/poison_train.csv")

# The test data to apply poisoning to
test_df = pd.read_csv("/var/lib/cdsw/share/test.csv")

# Apply poisoning to the training data
# I pass it the function to apply to all the images (draw white rectangle)
# I tell it to poison class 0 (bad roads)
# I want to save all images to 'data_poisoning/w_rect_2/'
apply_poisoning(poison_df, draw_rectangle, 0, "w_rect_2", 80)
# save the poison training df to a csv (so we can reload this)
poison_df.to_csv("data_poisoning/w_rect_2.csv")

# After running this function, you should now see a folder with the poisoned images

# Now we want to create an evaluation for this.
# Lets go ahead and poison the OPPOSITE class for our TEST data to create our
# poison test set (used to evaluate attack effectiveness)
# ALWAYS USE 100 with the test set!
apply_poisoning(test_df, draw_rectangle, 1, "w_rect_2_test", 100)
test_df.to_csv("data_poisoning/w_rect_2_test.csv")
# with this dataset, you should be able to take:
#   1 - accuracy on test set
# this will get you the confidence that you're able to impact the predictions given
# your poisoning methodology


# for this example, I don't map the bad roads to anything
# (no poisoning for the good roads in training data)
# therefore, they are unneeded in the poison test set, so lets delete them
drop_class(test_df, 0)

