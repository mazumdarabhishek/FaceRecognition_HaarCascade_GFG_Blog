import cv2

# consume live feed from web cam using opencv

vid_stream = cv2.VideoCapture(0)

while True:

    _, img = vid_stream.read()

    cv2.imshow("Web Cam", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release all streams and close window

vid_stream.release()
cv2.destroyAllWindows()