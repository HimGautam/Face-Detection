import cv2

# creating objects of class obtained from opencv lib
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')   
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# def detect fn
def detect(gray,frame):  
    face=face_cascade.detectMultiScale(gray, 1.3, 5) 
    # face return turple of x,y,width,height of face detected
   
    for (x,y,w,h) in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h),(255,0,0),2)
        # above line creates rectangle on the face in color image
       
        # when face is detected the region of interest is defined to look for eyes and smile
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
       
        # same process is repeated for eyes and smile
        eyes=eye_cascade.detectMultiScale(roi_gray,1.1,22)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
        
        smile=smile_cascade.detectMultiScale(roi_gray,1.7,22)
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0,0,255), 2)
    
    return frame

video_capture= cv2.VideoCapture(0)
# for turning on webcam

# infinite loop for live detection
while True:
    _,frame = video_capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #converting colored image frame to grayscale image
    canvas = detect(gray,frame)
    cv2.imshow('Video',canvas)
  
    if cv2.waitKey(1) & 0xFF == ord('q'):  #press "q" to break the infinite loop 
        break
video_capture.release()
cv2.destroyAllWindows()
