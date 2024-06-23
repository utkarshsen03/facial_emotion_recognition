# Numerical Analysis
import numpy as np
# Image processing
import cv2


class preprocessing():
    '''import image and apply haar-cascade on it to detect face.
    Reshape the image to fit in the model and then use model to predict the class.'''
    def __init__(self,path):
        '''path: Location of the image'''
        self.path = path
        
    def cascade(self):
        '''Face detection using haar cascade'''
        # Reading image
        img = cv2.imread(self.path)
        # conversion to gray image 
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # loading haar-cascade model
        face_cascade = cv2.CascadeClassifier(r'models/haarcascade_frontalface_default.xml')
        # Object for cascade classifier
        faces = face_cascade.detectMultiScale(gray_img, 1.2, 5)
        for (x, y, w, h) in faces:
            self.face = gray_img[y-10:y+h+20, x-10:x+w+10]
            
        return self.face
    
    def reshape(self):
        '''Reshaping face image'''
        resize_img = cv2.resize(self.face,(64,64))
        array = np.array(resize_img)
        self.reshape_img = array.reshape(-1,64, 64, 1)
        return self.reshape_img
    
    def standardize(self):
        '''Standardizing face image'''
        def rescale(data):
            return data/255
        self.output = np.array(list(map(rescale,self.reshape_img)))
        return self.output