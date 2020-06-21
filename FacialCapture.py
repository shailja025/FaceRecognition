import cv2
import numpy as np

# Loading HAARCASCASDE face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Function to detect face and give cropped face as output
def face_extractor(img):
    
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    
    if faces is ():
        return None
    
   
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

# Capturing frames using webcame <0 for default, 1 for external>
cap = cv2.VideoCapture(0)
count = 0


while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        
        # Save file in specified directory with unique name
        file_name_path = 'C://Users//HP//Desktop//MLOPS//' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
        
    else:
        print("Error finding face")
        pass

    if cv2.waitKey(1) == 13 or count == 100: 
        break
        
cap.release()
cv2.destroyAllWindows()      
print("Face Picture collected")
