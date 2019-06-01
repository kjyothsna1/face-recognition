
# coding: utf-8

# In[2]:


# Face Recognition

# Importing the libraries
import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Defining a function that will do the detections
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        print (faces.shape)
        print ("Number of faces detected: " + str(faces.shape[0]))
        print("Data Found")
        cv2.putText(frame, "Number of faces detected: " + str(faces.shape[0]), (10, 30),
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        FaceFileName = "face_" + str(y) + ".jpg"
        cv2.imwrite(FaceFileName, roi_color)
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return frame

# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _,frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()


# In[1]:


#import keras libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

model=Sequential()
model.add(Conv2D(32,3,3,input_shape=(64,64,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(output_dim=128,activation='relu',init='random_uniform'))
model.add(Dense(output_dim=1,activation='sigmoid',init='random_uniform'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

#x_train = train_datagen.flow_from_directory(r'C:\Users\iiitbasar\Desktop\Face detection\Face detection\training_se',target_size=(64,64),batch_size=32,class_mode='binary')
#x_test = train_datagen.flow_from_directory(r'C:\Users\iiitbasar\Desktop\Face detection\Face detection\test_set',target_size=(64,64),batch_size=32,class_mode='binary')
x_train = train_datagen.flow_from_directory(r'C:\Users\iiitbasar\Desktop\Face detection\Face detection\dataset\dataset\train set',target_size=(64,64),batch_size=32,class_mode='binary')
x_test = train_datagen.flow_from_directory(r'C:\Users\iiitbasar\Desktop\Face detection\Face detection\dataset\dataset\test set',target_size=(64,64),batch_size=32,class_mode='binary')

print(x_train.class_indices)

model.fit_generator(x_train,samples_per_epoch = 800,epochs=25,validation_data=x_test,nb_val_samples=8)

model.save('mymodel.h5')


# In[2]:


from keras.models import load_model
import numpy as np
import cv2
model =load_model(r'mymodel.h5')# give the model name to fetch the info

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


from skimage.transform import resize
def detect(frame):
    try:
        img= resize(frame,(64,64))
        img = np.expand_dims(img,axis=0)
        if(np.max(img)>1):
            img =img/255.0
        prediction =model.predict(img)
        print (prediction)
        prediction_class = model.predict_classes(img)
        print(prediction_class)
    except AttributeError:
        print("shape not found")
frame=cv2.imread(r"C:\Users\iiitbasar\Desktop\Face detection\Face detection\dataset\dataset\train set")
#frame= cv2.imread(r"C:\Users\iiitbasar\Desktop\Face detection\Face detection\training_set\varalaxmi") #select the directory path of that image to be uploaded
data= detect(frame)


# In[4]:


import numpy as np
from keras.preprocessing import image

from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os
from keras.models import load_model
classifier = load_model(r'mymodel.h5')# give the name of the file which they have saved after model prediction
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
root = Tk()
root.geometry("550x300+300+150")
root.resizable(width=True, height=True)

def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename
def open_img():
    x = openfn()
    test_image = image.load_img(x, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict_classes(test_image)
    print(result)
    index=["",""] #keep the names of the images trained in data set
    label = Label( root, text="Prediction : "+index[result[0][0]])
    label.pack()
    img = Image.open(x)
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.pack()

btn = Button(root, text='open image', command=open_img).pack()

root.mainloop()

