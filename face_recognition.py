import cv2
# import the xml file for face detection using openCV
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create a empty model and load the trained LBPH model
recognizer_model = cv2.face.LBPHFaceRecognizer_create()
recognizer_model.read('face_recognizer.xml')

# create a function to draw Bounding Box around the detected face
def draw_BBox(img, classifier, scale_factor, minmum_neighbors, color, model):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cord = classifier.detectMultiScale(img_gray, scale_factor, minmum_neighbors)
    

    cord = []
    for (x, y, w, h) in face_cord:

        cv2.rectangle(img= img, pt1= (x,y), pt2= (x+w, y+h), color= color, thickness=3)
        user_id, conf = model.predict(img_gray[y:y+h, x:x+w])
        print(user_id, conf)
        if user_id == 1 and conf >= 45:
            cv2.putText(img= img, text= str(user_id), org= (x, y-3), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6, color= (0,255,0), thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(img= img, text=str(round(conf,2)) , org= (x+70, y-3), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6, color= (0,255,0), thickness=1, lineType=cv2.LINE_AA)
        elif user_id == 2 and conf >=45:
            cv2.putText(img= img, text= str(user_id), org= (x, y-3), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6, color= (0,0,255), thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(img= img, text=str(round(conf,2)) , org= (x+70, y-3), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6, color= (0,0,255), thickness=1, lineType=cv2.LINE_AA)
      



def recognize_engine(img, model, faceDetector):

    _ = draw_BBox(img, faceDetector, 1.1, 10, (0,0,255), model)
    return img



# consume live feed from web cam using opencv

vid_stream = cv2.VideoCapture(0)

while True:

    _, img = vid_stream.read()

    img = recognize_engine(img,recognizer_model,faceDetector)
    cv2.imshow("Web Cam", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release all streams and close window

vid_stream.release()
cv2.destroyAllWindows()

