
# coding: utf-8

# In[1]:


#importing the keras libraries and packages
from keras.models import Sequential #its the model we are using 
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# In[3]:


#initialising the CNN(classifier neural network)
classifier=Sequential()
#Convolution
classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu')) #here input shape is the pixels of the image i.e 64*64 and 3 is that every pixel has 3 values. and the Conv2D is used to convert the image in 2D as it is in form of 64*64*3
#pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
#adding the second convolutional layer
classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))#pools the data from 3D to 2D


# In[4]:


#flattening
classifier.add(Flatten()) #SINCE WE ARE working with the 2D image which is actually the 3D image and by using flatten we are converting it to the one dimensional array
#Full connections
classifier.add(Dense(units=128,activation='relu')) # you can take any value at units we are taking 126(64+64) you may take 64+32 depending on the size of the data or image.
classifier.add(Dense(units=1,activation='sigmoid'))  #we are using here units=1 because we want to know whether it is a cat or dog sigmoid is used for yes or no answers.
  


# In[5]:


#compiling the CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])#optimiser is a reverse propogation whixh is used to handle or manages the values of weights if there comes errors.


# In[6]:


from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)


# In[7]:


training_set= train_datagen.flow_from_directory('Downloads/training_set',target_size=(64,64),batch_size=32,class_mode='binary')


# In[8]:


test_datagen=ImageDataGenerator(rescale = 1./255)
test_set=train_datagen.flow_from_directory('Downloads/test_set',target_size=(64,64),batch_size=32,class_mode='binary')


# In[9]:


classifier.fit_generator(training_set,steps_per_epoch=8000,epochs=25,validation_data=test_set,validation_steps=2000)#backpropogation 


# In[ ]:


# if the value of the accuracy goes down and the value_accuracy goes up then there is a bias which means that the model is memorising the things it is not making that what makes a dog a dog and a cat a cat.these is the beauty of the keras.just by  the values we can know what our model is doing. 


# In[53]:


#making out of sample prediction
import numpy as np
from keras.preprocessing import image
test_image=image.load_img('Downloads/single_prediction/cat_or_dog_1.jpg',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image, axis = 0)#making the image in a single array as axis=0
result=classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction='dog'
else:
    prediction='cat'
    
print(prediction)


# In[ ]:




