import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml");
font = cv2.FONT_HERSHEY_SIMPLEX

# id counter
id = 0
# id names, example => Reza: id=1,  etc
names = ['None', 'Reza'] 

# Initialize and start realtime video capture
camera = cv2.VideoCapture(0)
camera.set(3, 640) # set video width
camera.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*camera.get(3)
minH = 0.1*camera.get(4)

while(True):
    ret, frame =camera.read()
    # frame = cv2.flip(frame, -1) # Flip vertically
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5, minSize = (int(minW), int(minH)),)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # If confidence is less them 100 ==> "0" : perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(frame, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(frame, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',frame) 
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    
# cleanup
print("\n [INFO] Exiting Program...")
camera.release()
cv2.destroyAllWindows()