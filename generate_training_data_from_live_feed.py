
import cv2
# import the xml file for face detection using openCV

faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def create_train_data(img,user_name, user_id, img_id):

    cv2.imwrite(f"train_data/{user_name}_{user_id}_{img_id}.jpg", img)

# create a function to draw Bounding Box around the detected face
def draw_BBox(img, classifier, scale_factor, minmum_neighbors, color, text):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cord = classifier.detectMultiScale(img_gray, scale_factor, minmum_neighbors)

    cord = []
    for (x, y, w, h) in face_cord:

        cv2.rectangle(img= img, pt1= (x,y), pt2= (x+w, y+h), color= color, thickness=3)
        cv2.putText(img= img, text= text, org= (x, y-3), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.9, color= color, thickness=1, lineType=cv2.LINE_AA)
        cord = [x, y, w, h]
    
    return cord

def detect_face(img, faceDetector, img_id):
    red_color = (0,0,255)
    cord = draw_BBox(img, faceDetector, 1.1, 10, red_color, "Face")

    if len(cord)  == 4:
        face_img = img[cord[1]:cord[1]+cord[3], cord[0]:cord[0]+cord[2]]
        user_id = 2 # Hard coding for a sing user. This can be modularized for multiple users
        create_train_data(face_img,'Chai', user_id, img_id)

    return img



# consume live feed from web cam using opencv

vid_stream = cv2.VideoCapture(0)

img_id = 0

while True:

    _, img = vid_stream.read()

    img = detect_face(img, faceDetector, img_id)
    cv2.imshow("Web Cam", img)
    img_id += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release all streams and close window

vid_stream.release()
cv2.destroyAllWindows()

