import cv2

camera = cv2.VideoCapture(0)
camera.set(3, 640) # video width
camera.set(4, 480) # video height

# Cascade Classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# For each person, enter one numeric face id
face_id = input('\n enter user id =>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")

# Initialize individual sampling face count
count = 0
while(True):
    ret, frame = camera.read()
    # frame = cv2.flip(frame, -1) # flip video image vertically
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' +  
                    str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', frame)
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 50: # Take 30 face sample and stop video
         break

# cleanup
print("\n [INFO] Exiting Program...")
camera.release()
cv2.destroyAllWindows()