import cv2
import threading
import pathlib
import os
#C:\coding\face\database\trainer\trainer.yml
Face_Images = os.path.join(os.getcwd(), "dataset/lib")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C:\coding/face2\dataset/train/face-trainner.yml')
cascadePath = "C:\coding/face2\dataset\lib\haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX

# names = []
#C:\coding\face\database\face_resize
# path = 'C:\coding/face\databset/face_img'
# for path in pathlib.Path(path).iterdir():
#     names.append(path.name.split(".")[0].split("_")[0])

Face_Images = os.path.join(os.getcwd(), "dataset//face_img")
names = os.listdir(Face_Images)

class FaceDetect(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def get_frame(self):
        
        id = 0
        minW = 0.1*self.video.get(3)
        minH = 0.1*self.video.get(4)


        while True:
            img = self.frame
            #img = cv2.flip(img, -1) # Flip vertically
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale( 
                gray,
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize = (int(minW), int(minH)),
            )

            for(x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                
                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                # Check if confidence is less them 100 ==> "0" is perfect match
                if (confidence < 100):
                    if id > len(names) - 1:
                        continue
                    else:
                        id = names[id]
                        confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))
                
                cv2.putText(img, str(id), (x+5,y-5), font, 1, (0,0,255), 2)
                cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
                
            ret, jpeg = cv2.imencode('.jpg', img)
            return jpeg.tobytes()
        
    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()