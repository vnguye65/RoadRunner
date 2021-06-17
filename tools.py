from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from sklearn.model_selection import KFold

def print_cm(cm, labels):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels])
    # Print header
    print(" " * columnwidth, end="\t")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end="\t")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("%{0}s".format(columnwidth) % label1, end="\t")
        for j in range(len(labels)):
            print("%{0}d".format(columnwidth) % cm[i, j], end="\t")
        print()
        
        
def EvaluateModel (model, data, bs=32):
  pred = np.argmax(model.predict(data, batch_size = bs), axis=-1)
  #pred = np.argmax(pred, axis = 1)
  actual = data.classes
  cm = confusion_matrix(actual,pred)
  print('Confusion Matrix')
  print_cm(cm, ["0", "1", "2"])

  print('Classification Report')
  print(classification_report(actual, pred, target_names=["0", "1", "2"]))

"""
> class KerasClassifier(KerasClassifier):
>     
>     
>     def fit(self, x, y, **kwargs):
>        
>         if not issparse(x):
>             return super().fit(x, y, **kwargs)
>         
>         ############ adapted from KerasClassifier.fit   ######################   
>         if self.build_fn is None:
>             self.model = self.__call__(**self.filter_sk_params(self.__call__))
>         elif not isinstance(self.build_fn, types.FunctionType):
>             self.model = self.build_fn(
>                 **self.filter_sk_params(self.build_fn.__call__))
>         else:
>             self.model = self.build_fn(**self.filter_sk_params(self.build_fn))
> 
>         loss_name = self.model.loss
>         if hasattr(loss_name, '__name__'):
>             loss_name = loss_name.__name__
>         if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
>             y = to_categorical(y)
>         ### fit => fit_generator
>         fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit_generator))
>         fit_args.update(kwargs)
>         ############################################################
>         self.model.fit_generator(
>                     self.get_batch(x, y, self.sk_params["batch_size"]),
>                                         samples_per_epoch=x.shape[0],
>                                         **fit_args)                      
>         return self                               
> 
>     def get_batch(self, x, y=None, batch_size=32):
>         
>         index = np.arange(x.shape[0])
>         start = 0
>         while True:
>             if start == 0 and y is not None:
>                 np.random.shuffle(index)
>             batch = index[start:start+batch_size]
>             if y is not None:
>                 yield x[batch].toarray(), y[batch]
>             else:
>                 yield x[batch].toarray()
>             start += batch_size
>             if start >= x.shape[0]:
>                 start = 0
>   
>     def predict_proba(self, x):
>         
>         if not issparse(x):
>             return super().predict_proba(x)
>             
>         preds = self.model.predict_generator(
>                     self.get_batch(x, None, self.sk_params["batch_size"]), 
>                                                val_samples=x.shape[0])
>         return preds
""" 
def exponential_decay (lr0, s):
  def exponential_decay_fn (epoch):
    return lr0 * 0.1 ** (epoch/s)
  return eponential_decay_fn


def piecewise_constant_fn (epoch):
  if epoch < 5:
    return 0.01
  elif epoch < 15:
    return 0.005
  else:
    return 0.001