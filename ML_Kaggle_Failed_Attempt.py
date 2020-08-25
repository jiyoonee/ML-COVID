#!/usr/bin/env python
# coding: utf-8

# ### Load Data

# In[61]:


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


# In[278]:


import keras
from keras.models import Sequential 
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, ZeroPadding2D, Conv2D, MaxPooling2D,Input,SeparableConv2D, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator #Data augmentation and preprocessing
from keras.utils import to_categorical 
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
import cv2
from PIL import Image
from pathlib import Path
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,recall_score,confusion_matrix,classification_report


# In[386]:


path_train="/Users/jiyoon-song/Desktop/4771-sp20-covid/train"
path_test="/Users/jiyoon-song/Desktop/4771-sp20-covid/test"
path_val="/Users/jiyoon-song/Desktop/4771-sp20-covid/val"


# ### Classification

# In[495]:


from keras import applications
model = applications.VGG16(input_shape=(224,224,3), weights='imagenet', include_top=False)
top_model_weights_path = 'vgg.h5'
model.summary()


# In[488]:


from keras.applications.vgg16 import preprocess_input

train_gen = ImageDataGenerator(rescale=1./255,
                               preprocessing_function=preprocess_input,
                               shear_range = 0.2,
                               zoom_range = 0.2,
                               horizontal_flip = True)

val_gen = ImageDataGenerator(rescale=1./255,
                            preprocessing_function=preprocess_input)

train_batch = train_gen.flow_from_directory(path_train,
                                            target_size = (224, 224),
                                            classes = ["normal", "viral", "bacterial", "covid"],
                                            batch_size=32,
                                            class_mode = "categorical")
val_batch = val_gen.flow_from_directory(path_val,
                                        target_size = (224, 224),
                                        classes = ["normal", "viral", "bacterial", "covid"],
                                        batch_size=32,
                                        class_mode = "categorical")
test_batch = val_gen.flow_from_directory("/Users/jiyoon-song/Desktop/4771-sp20-covid",
                                         target_size = (224, 224),
                                         classes = ["test"],
                                         shuffle=False,
                                         class_mode = "categorical")

print(train_batch.image_shape)


# In[499]:


val_batch.filenames


# In[490]:


from keras.optimizers import SGD

layer_dict = dict([(layer.name, layer) for layer in model.layers])
x = layer_dict['block2_pool'].output 
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.7)(x)
preds=Dense(4, activation='softmax')(x)

'''
x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
preds=Dense(4,activation='softmax')(x)
'''

from keras.models import Model
custom_model = Model(input=model.input, output=preds)

for layer in custom_model.layers[:7]:
    layer.trainable = False

custom_model.compile(optimizer=SGD(learning_rate = 0.00001), loss='categorical_crossentropy',metrics=['accuracy'])


# In[491]:


custom_model.summary()


# In[492]:


train_batch.reset()
val_batch.reset()

checkpoint = ModelCheckpoint("vgg16_1.h5",
                             monitor='val_loss', 
                             verbose=1, 
                             save_best_only=True, 
                             save_weights_only=True, 
                             mode='auto', 
                             period=1)
early = EarlyStopping(monitor='val_loss', 
                      min_delta=0, 
                      patience=25, 
                      verbose=2, 
                      mode='auto')

reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.8,
    patience=5,
    verbose=1, 
    mode='auto',
    min_delta=0.0001, 
    cooldown=1, 
    min_lr=0.0001
)

history = custom_model.fit_generator(generator=train_batch,
                                     validation_steps=step_size_val,
                                     callbacks=[early_stop,checkpoint,reduce],
                                     validation_data = val_batch,
                                     steps_per_epoch=step_size_train,
                                     shuffle=False,
                                     epochs=100)
custom_model.save_weights("vgg16_1.h5")


# In[468]:


import matplotlib.pyplot as plt 

print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
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


# In[469]:


step_size_test=(test_batch.n//test_batch.batch_size) + 1
test_batch.reset()
pred = custom_model.predict_generator(test_batch,
                                      steps=step_size_test,
                                      verbose=1)


# In[470]:


pred


# In[471]:


predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_batch.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]


# In[472]:


print(train_batch.class_indices)
predictions


# In[476]:


filenames=test_batch.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results


# In[474]:


sorted_res = pd.DataFrame(columns = ['Id', 'label'])
for i in range(len(results)):
    filename = results.iloc[[i]]['Filename']
    ind = int(re.findall('\d+', str(filename))[1])
    sorted_res.loc[ind] = [ind, predictions[i]]
    #results.loc[i]['Filename'] = float(ind)
sorted_res.sort_values(['Id'], inplace=True)
sorted_res


# In[475]:


sorted_res.to_csv('submission.csv', index=False)


# In[478]:


results.columns = ['Id', 'label']


# In[479]:


results.to_csv('submission.csv', index=False)


# In[ ]:




