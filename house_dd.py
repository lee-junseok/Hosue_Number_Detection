#!/usr/bin/env python
# coding: utf-8

# ### Data from https://www.kaggle.com/c/digit-recognizer

# Go to [Data Preparation](#data)

# Go to [CNN](#CNN)

# Go to [Evaluation](#evaluation)

# # Digit Recognizer using TensorFlow Keras

# In[2]:


import warnings
warnings.filterwarnings("ignore")

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# ### No GPU on this laptop :(

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

sns.set(style='white', context='notebook', palette='deep')


# <a id='data' />

# # Data preparation

# In[4]:


from scipy.io import loadmat


# In[361]:


mat = loadmat('train_32x32.mat')



import cv2


# In[ ]:




def mask_side(img):
    width = 32
    height = 32
    kernel_size = max(1,int(np.random.normal(loc=9, scale=2, size=None)))
    kernel_size = kernel_size+ (kernel_size+1)%2 #kernel size must be odd
    blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    mask = np.zeros((width, height, 3), dtype=np.uint8)
    mask = cv2.rectangle(mask, (int(width/4), 0), (int(width*3/4), height), (255, 255, 255), -1)
    out = np.where(mask==(255, 255, 255), img, blurred_img)
    return out


# In[322]:


def display_preprocessed_samples(n_of_samples):
    """ This function shows 6 images with their predicted and real labels"""
    sample_idx_arr = np.random.choice(range(len(mat['y'])),n_of_samples)
    n = 0
    nrows = int(np.sqrt(len(sample_idx_arr)))
    ncols = int(np.ceil(len(sample_idx_arr)/nrows))
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True, figsize = (nrows*3,ncols*2))
    plt.subplots_adjust(hspace = 0.6)
    for row in range(nrows):
        for col in range(ncols):
            if (row)*(ncols)+(col+1) > len(sample_idx_arr): return
            sample_i = sample_idx_arr[n]
            preprocessed_img = mask_side(mat['X'][:,:,:,sample_i])
            ax[row,col].imshow(preprocessed_img)
            ax[row,col].set_title("Index : {}\nLabel : {}".format(sample_i,mat['y'][sample_i]))
            ax[row,col].axis("off")
            n += 1



from tqdm import tqdm


# Gray Scale: 0.2989 R + 0.5870 G + 0.1140 B




# - 32X32 pixels

# ## Train set

# In[ ]:


mat = loadmat('train_32x32.mat')
preprocessed_set = np.zeros((mat['X'].shape[3],mat['X'].shape[0],mat['X'].shape[1]),dtype=np.uint8)
for i in tqdm(range(len(mat['y']))):
    preprocessed_img = mask_side(mat['X'][:,:,:,i])
    preprocessed_set[i,...] = np.dot(preprocessed_img,[0.2989, 0.5870, 0.1140])
df = pd.DataFrame(np.hstack((preprocessed_set.reshape(len(mat['y']),-1), mat['y'])))
df.rename(columns={1024:'label'}, inplace = True)
trainY = df["label"]
trainX = df.drop(labels = ["label"],axis = 1)
del df

g = sns.countplot(trainY)

trainY.value_counts()


# ## Test set

# In[305]:


mat = loadmat('test_32x32.mat')
preprocessed_set = np.zeros((mat['X'].shape[3],mat['X'].shape[0],mat['X'].shape[1]),dtype=np.uint8)
for i in tqdm(range(len(mat['y']))):
    preprocessed_img = mask_side(mat['X'][:,:,:,i])
    preprocessed_set[i,...] = np.dot(preprocessed_img,[0.2989, 0.5870, 0.1140])
df = pd.DataFrame(np.hstack((preprocessed_set.reshape(len(mat['y']),-1), mat['y'])))
df.rename(columns={1024:'label'}, inplace = True)
testY = df["label"]
testX = df.drop(labels = ["label"],axis = 1)
del df

g = sns.countplot(testY)

testY.value_counts()


# ## Missing values

# In[306]:





# Normalize the data
trainX = trainX / trainX.max().max() #258
testX = testX / trainX.max().max()


# ## Reshape

# In[334]:


# Reshape image in 3 dimensions (height = 28px, width = 28px , channel = 1)
trainX = trainX.values.reshape(-1,32,32,1)
testX = testX.values.reshape(-1,32,32,1)


# - grayscaled images, hence one channel

# ## Label encoding

# In[348]:


trainY = np.where(trainY==10,0,trainY)
testY = np.where(testY==10,0,testY)


# In[349]:


# labels to one hot vectors
trainY = to_categorical(trainY, num_classes = 10)
testY = to_categorical(testY, num_classes = 10)




# ## Split training and valdiation set

# In[352]:


# Split the train and the validation set for the fitting
trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size = 0.1)



# <a id='CNN' />

# # CNN
# ## Modeling

# In[354]:


#[[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu', input_shape = (32,32,1))) #filters = num of filters, kernel_size = filter size
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

#fully connected
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


# - ## RMSprop Optimizer - not used here

# In[56]:


optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)


# ### Compile

# In[63]:


model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# ### Annealer

# In[58]:


# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)


# - ## Adam Optimizer - no need annealer

# In[355]:


optimizer = Adam(lr=0.001,beta_1=0.9,beta_2=0.999)


# ### compile

# In[356]:


model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# ## Epochs and Batch Size

# In[373]:


epochs = 30
batch_size = 86


# ## Augmentation

# In[374]:


# prevent overfitting
datagen = ImageDataGenerator(
        rotation_range=10,  # degree
        zoom_range = 0.1,
        width_shift_range=0.1,
        height_shift_range=0.1)


datagen.fit(trainX)


# - ### Adams

# In[ ]:
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint("best_model.hdf5", monitor='loss', verbose=0,
    save_best_only=True, mode='auto', save_freq=1)

# Fit the model
history = model.fit_generator(datagen.flow(trainX,trainY,batch_size=batch_size),
                             epochs=epochs,validation_data=(valX,valY),
                              verbose = 2, steps_per_epoch=trainX.shape[0]//batch_size,
                              callbacks=[checkpoint])


import pickle
with open('trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
model.save('keras_model_save')
import keras
model_load = keras.models.load_model('keras_model_save')
# <a id='evaluation' />

# # Evaluation
# ## Training and validation curves

# - ### Adams

# In[72]:


# # Plot the loss and accuracy curves for training and validation
# fig, ax = plt.subplots(2,1,figsize = (14,8))
# ax[0].plot(history.history['loss'], color='b', label="Training loss")
# ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
# legend = ax[0].legend(loc='best', shadow=True)

# ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
# ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
# legend = ax[1].legend(loc='best', shadow=True)


# # ## Confusion matrix

# # In[81]:


# # confusion matrix
# # Predict the values from the validation dataset
# Y_pred = model.predict(valX)
# # Convert predictions classes to one hot vectors
# Y_pred_classes = np.argmax(Y_pred,axis = 1)
# # Convert validation observations to one hot vectors
# Y_true = np.argmax(valY,axis = 1)
# # compute the confusion matrix
# confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
# plt.figure(figsize=(10,9))
# sns.heatmap(confusion_mtx, cmap='RdBu_r', annot=True, fmt= 'd', center=0.0)
# plt.title("Confusion Matrix",fontsize=14)
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.show()


# # ## Errors

# # In[113]:


# # Display some error results

# # Errors are difference between predicted labels and true labels
# errors = (Y_pred_classes - Y_true != 0)

# Y_pred_classes_errors = Y_pred_classes[errors]
# Y_pred_errors = Y_pred[errors]
# Y_true_errors = Y_true[errors]
# valX_errors = valX[errors]

# def display_errors(errors_index,img_errors,pred_errors, obs_errors):
#     """ This function shows 6 images with their predicted and real labels"""
#     n = 0
#     nrows = int(np.sqrt(len(errors_index)))
#     ncols = int(np.ceil(len(errors_index)/nrows))
#     fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True, figsize = (nrows*3,ncols*2))
#     plt.subplots_adjust(hspace = 0.6)
#     for row in range(nrows):
#         for col in range(ncols):
#             if (row)*(ncols)+(col+1) > len(errors_index): return
#             error = errors_index[n]
#             ax[row,col].imshow((img_errors[error]).reshape((28,28)))
#             ax[row,col].set_title("Predicted :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
#             ax[row,col].axis("off")
#             n += 1

# # Probabilities of the wrong predicted numbers
# Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)

# # Predicted probabilities of the true values in the error set
# true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# # Difference between the probability of the predicted label and the true label
# delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# # Sorted list of the delta prob errors
# sorted_dela_errors = np.argsort(delta_pred_true_errors)

# # Top n errors
# most_important_errors = sorted_dela_errors[-12:]

# # Show the top n errors
# display_errors(most_important_errors, valX_errors, Y_pred_classes_errors, Y_true_errors)

