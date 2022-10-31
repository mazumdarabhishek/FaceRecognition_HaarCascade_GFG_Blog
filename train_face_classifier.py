# DEPENDENCY FOR IMPORTING FACE MODULE 
# pip install opencv-contrib-python


import cv2
import glob
from PIL import Image
import os
import numpy as np



# Function to train  image recognition model
def train_recog_model(img_dir_path:str):
    # get all the image paths 
    img_paths = glob.glob(img_dir_path+'/*')
    

    face = []
    user_id = []
    # Loop over each image in the train_data folder
    for image in img_paths:
        img = Image.open(image).convert('L')
        # Convert image to numpy array
        numpy_image =  np.array(img, 'uint8')
        # Extract user_id from image file name 
        idx = int(image.split('''\\''')[-1].split('_')[1])

        # Append numpy image array to face list and its corresponding user_id to user_id list
        face.append(numpy_image)
        user_id.append(idx)
    
    # convert user id list to a numpr array  
    user_id = np.array(id)
    
    #  Initiate LBPHFaceRecognizer model and train the model by passing image array and its corresponding id array
    # Save it to a xml file called face_recognizer.xml
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(face, id)
    model.write('face_recognizer.xml')

# Call the function and pass the data directory name 
train_recog_model("train_data")