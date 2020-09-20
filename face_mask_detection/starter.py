import cv2, os
import matplotlib.pyplot as plt
import math
#from PIL import Image
import numpy as np
from keras.utils import np_utils
#####
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from tensorflow.python.keras import models

#load data
#convert to grayscale
#normalize it 
#make sure that your target values are 0 and 1 (one dimentional)
#start training (take the conv2d and start off with a low number of channels)
# do another layer with a higher number of channels
# flatten it
# pass it to dense layers 
# output layer= softmax, sigmoid
# Classify Accuracy

def display_image(im,dpi = 160):
    height, width= im.shape[0], im.shape[1]
    #size = width / float(dpi), height / float(dpi)
    fig, ax = plt.subplots(figsize=(width,height), subplot_kw={'xticks':[], 'yticks':[]})
    ax.imshow(im, cmap='gray')
    
def mask_states():
    file = open('Training_set_face_mask.csv','r')
    line = file.readline()
    line = file.readline().rstrip('\n').split(',')
    mask_state = []
    while line != ['']:
        target = 0
        if line[1] == 'without mask':
            target = 1
        mask_state += [target]
        line = file.readline().rstrip('\n').split(',')
    file.close()
    return mask_state

def images_list():
    data_path = 'train'
    #img_size = 100
    data = []
    smallest = math.inf
    for image_name in os.listdir(data_path):
        #print(image_name)
        image_path = os.path.join(data_path, image_name)
        img = cv2.imread(image_path)
        if img.shape[0] < smallest:
            smallest = img.shape[0]
        try:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #display_image(gray)
            data.append(cv2.resize(gray,(25,25)))
            #data.append(gray)
        except Exception as e:
            print('Exception: ', e)
        #break;
    #print(smallest)
    return data

'''
# DATA PROCESSING  
'''

# label_dict = {'with mask':0, 'without mask':1}
# categories = ['with mask', 'without mask']
# labels = [0,1]
# test = os.listdir('train')

#data = images_list()
data = np.array(images_list())/255.0
data = np.reshape(data,(data.shape[0],25,25,1))
target = np.array(mask_states())
target = np_utils.to_categorical(target) #keep this janeth won't
#do the mean of target and if its 1 something is busted


'''
# CONVOLUTIONAL ARCHITECTURE
'''

model = Sequential()
# first layer
model.add(Conv2D(32,(1,1), input_shape=(25, 25,1)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))

#second layer
model.add(Conv2D(64,(3,3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2,2,)))

# flatten 
#model.add(Flatten())
print(model.output_shape)


# dense layer 
model.add(Dense(67712))
model.add(Dropout(0.5))
model.add(Dense(1,activation='softmax'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# Training data?? 
#train_data,train_target = (data, target)
train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)
checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')

'''
# TRAINING 
'''

history = model.fit(data,target,epochs=10,callbacks=[checkpoint],validation_split=0.2)
#history = model.fit(train_data,train_target,epochs=20,callbacks=[checkpoint],validation_split=0.2)

plt.plot(history.history['loss'],'r',label='training loss')
plt.plot(history.history['val_loss'],label='validation loss')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'],'r',label='training accuracy')
plt.plot(history.history['val_accuracy'],label='validation accuracy')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

print(model.evaluate(test_data,test_target))




















