# FaceRecognition_HaarCascade_GFG_Blog

### This repository is the code for GeekforGeeks blog on *How to create Face Recognition Using Haar Cascade*

#### Dependency
      pip install numpy
      pip install opencv-python
      pip install opencv-contrib-python
      pip install Pillow 

-  ***FaceDetectionCascade.py*** is the code for running face detection using OpenCV.
-  ***OpenCV_videoCapture.py*** gives a becis code snippet to help beginners in using opencv 
-  ***generate_training_data_from_live_feed.py*** is the code to get training samples from a live feed, the default is set to 0 i.e., video cam of a computer
-  ***train_face_classifier.py*** is used to train a LBPH classifier using the training images captured
-  ***face_recognition.py*** is the code to consume the  LBHP model created to recognize images in realtime video feed.
