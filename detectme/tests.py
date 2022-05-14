from django.test import TestCase
import os
import numpy as np
import cv2
# Create your tests here.

Face_Images = os.path.join(os.getcwd(), "face2\dataset/face_img/sanghoon")
#print(Face_Images)
# labels = os.listdir(Face_Images)
# print(labels)

cap = cv2.VideoCapture(0) # 노트북 웹캠을 카메라로 사용
cap.set(3,640) # 너비
cap.set(4,480) # 높이

i = 126
while True:
    ret, cam = cap.read()

    if(ret):
        cv2.imshow('camera', cam)
        if cv2.waitKey(1) == ord("q"):
            ret, frame = cap.read() # 사진 촬영
            frame = cv2.flip(frame, 1) # 좌우 대칭
            name = Face_Images + "/" + str(i) + ".jpg"
            i += 1
            cv2.imwrite(name, frame) # 사진 저장
        elif cv2.waitKey(1) & 0xFF == 27: # esc 키를 누르면 닫음
            break

cap.release()
cv2.destroyAllWindows()
