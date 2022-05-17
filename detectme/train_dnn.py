import cv2, dlib
import numpy as np
import os
from PIL import Image

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('face2/dataset/lib/shape_predictor_68_face_landmarks_GTX.dat')
facerec = dlib.face_recognition_model_v1("face2/dataset/lib/dlib_face_recognition_resnet_model_v1.dat")


Face_ID = -1 
pev_person_name = ""
y_ID = []
x_train = []

Face_Images = os.path.join(os.getcwd(), "face2/dataset/face_img") #이미지 폴더 지정
print(Face_Images)
win = dlib.image_window()
for root, dirs, files in os.walk(Face_Images) : #파일 목록 가져오기
    for file in files :
        if file.endswith("jpeg") or file.endswith("jpg") or file.endswith("png") : #이미지 파일 필터링
            path = os.path.join(root, file)
            person_name = os.path.basename(root)
            #print(path, person_name)
 
            if pev_person_name != person_name : #이름이 바뀌었는지 확인
                Face_ID=Face_ID+1
                pev_person_name = person_name
            
            img = cv2.imread(path) #이미지 파일 가져오기
            #gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            win.clear_overlay()
            win.set_image(img)

            dets = detector(img, 1)
            print("Number of faces detected: {}".format(len(dets)))

            # Now process each face we found.
            for k, d in enumerate(dets):
                print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                    k, d.left(), d.top(), d.right(), d.bottom()))
                # Get the landmarks/parts for the face in box d.
                shape = sp(img, d)
                win.clear_overlay()
                win.add_overlay(d)
                win.add_overlay(shape)
            # dlib_shape = recognizer(gray_image, face)
            # shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])
            # top_left = np.min(shape_2d, axis=0)
            # bottom_right = np.max(shape_2d,axis=0)
            # center_X, center_Y = np.mean(shape_2d, axis=0).astype(np.int)
            
            
                face_descriptor = facerec.compute_face_descriptor(img, shape)
                print(face_descriptor)
            # for i in shape_2d:
            #     cv2.circle(img,center = tuple(i), radius =1, color=(255,255,255),thickness=2, lineType=cv2.LINE_AA)

            #     cv2.circle(img, center=tuple(top_left), radius=1, color=(255, 1, 1), thickness=2, lineType=cv2.LINE_AA)
            #     cv2.circle(img, center=tuple(bottom_right), radius=1, color=(255, 1, 1), thickness=2, lineType=cv2.LINE_AA)
            #     cv2.circle(img, center=tuple((center_X, center_Y)), radius=1, color=(1, 1, 255), thickness=2, lineType=cv2.LINE_AA)

            #     cv2.imshow('img', img)
            #     cv2.waitKey(0)
            
            print("Computing descriptor on aligned image ..")
        
            # Let's generate the aligned image using get_face_chip
            face_chip = dlib.get_face_chip(img, shape)        

            # Now we simply pass this chip (aligned image) to the api
            face_descriptor_from_prealigned_image = facerec.compute_face_descriptor(face_chip)                
            print(face_descriptor_from_prealigned_image)        
            
            dlib.hit_enter_to_continue()
            