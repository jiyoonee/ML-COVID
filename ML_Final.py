#!/usr/bin/env python
# coding: utf-8

# ### Load Data

# In[17]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import random


# In[2]:


os.listdir("/Users/jiyoon-song/Desktop/4771-sp20-covid")


# In[60]:


path_train="/Users/jiyoon-song/Desktop/4771-sp20-covid/train/train"
path_test="/Users/jiyoon-song/Desktop/4771-sp20-covid/test/test"
csv_train= "/Users/jiyoon-song/Desktop/4771-sp20-covid/train.csv"
csv_test= "/Users/jiyoon-song/Desktop/4771-sp20-covid/test.csv"


# In[61]:


csv_train = pd.read_csv(csv_train)


# ### Preprocessing

# In[91]:


# train: 1127 samples
# test: 484 samples     
# train test: 70 30
# train val: 80 20
# --> validation: 225

# train val test: 902 225 484


# In[97]:


#DO NOT RUN AGAIN
for i in range(225):
    filename = random.choice(os.listdir(path_train))
    os.rename(path_train + "/" +filename, "/Users/jiyoon-song/Desktop/4771-sp20-covid/val/" + filename)


# In[104]:


for filename in os.listdir(path_train):
    print(filename)
    ind = int(re.findall('\d+', filename)[0])
    if (csv_train.iloc[[ind]]['label'] == 'normal').any():
        os.rename(path_train + "/" +filename, "/Users/jiyoon-song/Desktop/4771-sp20-covid/train/normal/" + filename)
    elif (csv_train.iloc[[ind]]['label'] == 'viral').any():
        os.rename(path_train + "/" +filename, "/Users/jiyoon-song/Desktop/4771-sp20-covid/train/viral/" + filename)
    elif (csv_train.iloc[[ind]]['label'] == 'bacterial').any():
        os.rename(path_train + "/" +filename, "/Users/jiyoon-song/Desktop/4771-sp20-covid/train/bacterial/" + filename)
    elif (csv_train.iloc[[ind]]['label'] == 'covid').any():
        os.rename(path_train + "/" +filename, "/Users/jiyoon-song/Desktop/4771-sp20-covid/train/covid/" + filename)


# In[109]:


path_val = "/Users/jiyoon-song/Desktop/4771-sp20-covid/val/"
for filename in os.listdir(path_val):
    if os.path.isdir(path_val + filename):
        continue
    ind = int(re.findall('\d+', filename)[0])
    if (csv_train.iloc[[ind]]['label'] == 'normal').any():
        os.rename(path_val +filename, path_val + "normal/" + filename)
    elif (csv_train.iloc[[ind]]['label'] == 'viral').any():
        os.rename(path_val +filename, path_val + "viral/"  + filename)
    elif (csv_train.iloc[[ind]]['label'] == 'bacterial').any():
        os.rename(path_val +filename, path_val + "bacterial/" + filename)
    elif (csv_train.iloc[[ind]]['label'] == 'covid').any():
        os.rename(path_val +filename, path_val + "covid/"  + filename)


# In[46]:


import keras
from keras.models import Sequential 
from keras.models import Model
from keras.layers import LeakyReLU, Dense, Dropout, Flatten, ZeroPadding2D, Conv2D, MaxPooling2D,Input,SeparableConv2D, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator #Data augmentation and preprocessing
from keras.utils import to_categorical 
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
import cv2
import datetime
import math
from PIL import Image
from pathlib import Path
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,recall_score,confusion_matrix,classification_report


# In[4]:


path_train="/Users/jiyoon-song/Desktop/4771-sp20-covid/train"
path_test="/Users/jiyoon-song/Desktop/4771-sp20-covid/test"
path_val="/Users/jiyoon-song/Desktop/4771-sp20-covid/val"


# In[149]:


from keras import applications
from keras.applications.vgg16 import preprocess_input

vgg16 = applications.VGG16(include_top=False, weights='imagenet')
train_datagen = ImageDataGenerator(rescale=1. / 255,
                             shear_range = 0.2,
                             rotation_range = 20,
                             zoom_range = 0.2,
                             horizontal_flip = True,
                             preprocessing_function=preprocess_input) 

datagen = ImageDataGenerator(rescale=1. / 255,
                             preprocessing_function=preprocess_input)


# In[7]:


img_width, img_height = 224, 224 
top_model_weights_path = 'vgg16.h5'
epochs = 100
batch_size = 32 


# In[18]:


#ONLY RUN ONCE
start = datetime.datetime.now()
 
train_gen = datagen.flow_from_directory( 
    path_train, 
    target_size=(img_width, img_height), 
    batch_size=batch_size, 
    class_mode=None, 
    shuffle=False) 
 
nb_train_samples = len(train_gen.filenames) 
num_classes = len(train_gen.class_indices) 
 
predict_size_train = int(math.ceil(nb_train_samples / batch_size)) 
 
bottleneck_features_train = vgg16.predict_generator(train_gen, predict_size_train) 
 
np.save('bottleneck_features_train.npy', bottleneck_features_train)
end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)


# In[19]:


#ONLY RUN ONCE
start = datetime.datetime.now()
 
val_gen = datagen.flow_from_directory( 
    path_val, 
    target_size=(img_width, img_height), 
    batch_size=batch_size, 
    class_mode=None, 
    shuffle=False) 
 
nb_val_samples = len(val_gen.filenames) 
 
predict_size_val = int(math.ceil(nb_val_samples / batch_size)) 
 
bottleneck_features_val = vgg16.predict_generator(val_gen, predict_size_val) 
 
np.save('bottleneck_features_val.npy', bottleneck_features_val)
end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)


# In[20]:


#ONLY RUN ONCE
start = datetime.datetime.now()
 
test_gen = datagen.flow_from_directory( 
    path_test, 
    target_size=(img_width, img_height), 
    batch_size=batch_size, 
    class_mode=None, 
    shuffle=False) 
 
nb_test_samples = len(test_gen.filenames) 
 
predict_size_test = int(math.ceil(nb_test_samples / batch_size)) 
 
bottleneck_features_test = vgg16.predict_generator(test_gen, predict_size_test) 
 
np.save('bottleneck_features_test.npy', bottleneck_features_test)
end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)


# ### Training

# In[150]:


gen_train_top = train_datagen.flow_from_directory( 
   path_train, 
   target_size=(img_width, img_height), 
   batch_size=batch_size, 
   class_mode='categorical', 
   shuffle=True) 
 
train_data = np.load('bottleneck_features_train.npy') 
 
train_labels = gen_train_top.classes 
  
train_labels = to_categorical(train_labels, num_classes=num_classes)


# In[151]:


gen_val_top = datagen.flow_from_directory( 
   path_val, 
   target_size=(img_width, img_height), 
   batch_size=batch_size, 
   class_mode='categorical', 
   shuffle=True) 
 
val_data = np.load('bottleneck_features_val.npy') 
 
val_labels = gen_val_top.classes 
 
val_labels = to_categorical(val_labels, num_classes=num_classes)


# In[152]:


gen_test_top = datagen.flow_from_directory( 
   path_test, 
   target_size=(img_width, img_height), 
   batch_size=batch_size, 
   class_mode='categorical', 
   shuffle=False) 
 
test_data = np.load('bottleneck_features_test.npy') 
 
test_labels = gen_test_top.classes 
 
test_labels = to_categorical(test_labels, num_classes=num_classes)


# In[153]:


start = datetime.datetime.now()

model = Sequential() 
model.add(Flatten(input_shape=train_data.shape[1:])) 
model.add(Dense(100, activation=keras.layers.LeakyReLU(alpha=0.3))) 
model.add(Dropout(0.7)) 
model.add(Dense(50, activation=keras.layers.LeakyReLU(alpha=0.3))) 
model.add(Dropout(0.7)) 
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.0001),
              metrics=['acc'])

early = EarlyStopping(monitor='val_loss', 
                      min_delta=0, 
                      patience=25, 
                      verbose=2, 
                      mode='auto')

history = model.fit(train_data, train_labels, 
                    epochs=100,
                    batch_size=batch_size, 
                    validation_data=(val_data, val_labels))

model.save_weights(top_model_weights_path)
eval_loss, eval_accuracy = model.evaluate(val_data, 
                                          val_labels, 
                                          batch_size=batch_size, 
                                          verbose=1)

print("Accuracy:", eval_accuracy) 
print("Loss:", eval_loss) 

end= datetime.datetime.now()
elapsed= end-start
print ("Time: ", elapsed)


# In[215]:


model.summary()


# In[154]:


import matplotlib.pyplot as plt 

print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Test set'], loc='upper left')
plt.show()

#Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Test set'], loc='upper left')
plt.show()


# In[166]:


preds = np.round(model.predict(test_data),0)
preds


# In[156]:


predicted_class_indices=np.argmax(preds,axis=1)
labels = (train_gen.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]


# In[164]:


predictions


# In[158]:


filenames=test_gen.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results


# In[159]:


sorted_res = pd.DataFrame(columns = ['Id', 'label'])
for i in range(len(results)):
    filename = results.iloc[[i]]['Filename']
    ind = int(re.findall('\d+', str(filename))[1])
    sorted_res.loc[ind] = [ind, predictions[i]]
sorted_res.sort_values(['Id'], inplace=True)
sorted_res


# In[160]:


sorted_res.to_csv('submission.csv', index=False)


# In[171]:


val_preds = np.round(model.predict(val_data),0)
val_preds


# In[172]:


val_labels


# In[176]:


classification_metrics = classification_report(val_labels, val_preds, target_names = ['bacterial', 'covid', 'normal', 'viral'])
print(classification_metrics)


# In[178]:


predicted_val_indices=np.argmax(val_preds,axis=1)
labels = (val_gen.class_indices)
labels = dict((v,k) for k,v in labels.items())
val_preds = [labels[k] for k in predicted_val_indices]
val_preds


# In[179]:


val_indices = np.argmax(val_labels, axis=1)
val_labels = [labels[k] for k in val_indices]
val_labels


# In[188]:


val_preds = np.array(val_preds)
val_labels = np.array(val_labels)


# In[214]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
ax= plt.subplot()
labels=['bacterial', 'covid', 'normal', 'viral']
matrix = confusion_matrix(val_labels, val_preds, labels)
sns.heatmap(matrix, annot=True, cmap="Blues")

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.xaxis.set_ticklabels(['bacterial', 'covid', 'normal', 'viral']); ax.yaxis.set_ticklabels(['bacterial', 'covid', 'normal', 'viral']);


# In[ ]:




