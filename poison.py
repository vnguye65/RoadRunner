## Imports

from PIL import Image
import imageio
from PIL import ImageDraw
import matplotlib.pyplot as plt
import cv2
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import os

#--------------------------------

poison_df = pd.read_csv("/var/lib/cdsw/share/poison_train.csv")
test_df = pd.read_csv("/var/lib/cdsw/share/test.csv")

#--------------------------------

  
def ReduceQuality (path:str, new_path:str):
  '''Reduce image to 2 bits '''
  image = imageio.imread(path)
  image = image & 0xC0
  imageio.imsave(new_path, image)
  #return image


def DownSampling (path:str, new_path = str):
  '''Downsampling by a factor of 8'''
  image = imageio.imread(path)
  image = image[0::6,0::6]
  imageio.imsave(new_path, image)
  #return image
  
  
# EXAMPLE: Drawing a white rectangle
def draw_rectangle(path, new_path):
  image = Image.open(path)
  im_draw = ImageDraw.Draw(image)
  im_draw.rectangle((2, 2, 12, 12), fill="white")
  image.save(new_path)
  
  

def RemoveMarkings (path, new_path: str, new_color = 100):
  image = cv2.imread(path)
  thresh = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY)[1]
  ind = (thresh == 255).nonzero()
  image[ind] = new_color
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  imageio.imsave(new_path, image)
  #return image_rgb
  

def AddNoise (path: str):
  np.random.seed(42)
  image = Image.open(path)
  noisy = np.random.normal(image)
  pca = PCA(3).fit_transform(noisy)
  components = pca.transform(noisy)
  filtered = pca.inverse_transform(components)
  return filter
  
  
for i in range(len(ind[0])):
  
  x = ind[0][i]
  y = ind[1][i]
  c = ind[2][i]
  color = image[x, y, c]
  
  if x == 0 and y == 0:
    surr = image[x:x+3, y:y+3, c].reshape(1, -1)
    #surr = np.delete(surr, np.where(surr >= color))
    #new = np.average(surr)
    new = np.random.choice(surr[0])
  elif x == 0:
    surr = image[x:x+3, y-1:y+2, c].reshape(1, -1)
    #surr = np.delete(surr, np.where(surr >= color))
    #new = np.average(surr)
    new = np.random.choice(surr[0])
  elif y == 0:
    surr = image[x-1:x+2, y:y+3, c].reshape(1, -1)
    #surr = np.delete(surr, np.where(surr >= color))
    #new = np.average(surr)
    new = np.random.choice(surr[0])
  else:
    surr = image[x-1:x+2, y-1:y+2, c].reshape(1, -1)
    #surr = np.delete(surr, np.where(surr >= color))
    #new = np.average(surr)
    new = np.random.choice(surr[0])

  if new > 0:
    image[x, y, c] = int(new)





def find_dominant_color(filename):
    #Resizing parameters
    #width, height = 150,150
    image = Image.open(filename)
    image = image.resize((width, height),resample = 0)
    #Get colors from image object
    pixels = image.getcolors(width * height)
    #Sort them by count number(first element of tuple)
    sorted_pixels = sorted(pixels, key=lambda t: t[0])
    #Get the most frequent color
    dominant_color = sorted_pixels[-1][1]
    return dominant_color


  
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
  if not os.path.exists('data_poisoning_GMU'):
    os.makedirs('data_poisoning_GMU')
  
  # make folder
  outpath = 'data_poisoning_GMU/' + name
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



# Apply poisoning to the training data
# I pass it the function to apply to all the images (draw white rectangle)
# I tell it to poison class 0 (bad roads)
# I want to save all images to 'data_poisoning/w_rect_2/'
apply_poisoning(poison_df, RemoveMarkings, 1, "no_road_markings", 80)
# save the poison training df to a csv (so we can reload this)
poison_df.to_csv("data_poisoning_GMU/no_road_markings.csv")

# After running this function, you should now see a folder with the poisoned images

# Now we want to create an evaluation for this.
# Lets go ahead and poison the OPPOSITE class for our TEST data to create our
# poison test set (used to evaluate attack effectiveness)
# ALWAYS USE 100 with the test set!
apply_poisoning(test_df, DownSampling, 1, "down_sampling", 80)
test_df.to_csv("data_poisoning_GMU/down_sampling_test.csv")
# with this dataset, you should be able to take:
#   1 - accuracy on test set
# this will get you the confidence that you're able to impact the predictions given
# your poisoning methodology


# for this example, I don't map the bad roads to anything
# (no poisoning for the good roads in training data)
# therefore, they are unneeded in the poison test set, so lets delete them
drop_class(test_df, 0)