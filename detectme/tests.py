from django.test import TestCase
import os
# Create your tests here.

Face_Images = os.path.join(os.getcwd(), "face2\dataset/face_img")
labels = os.listdir(Face_Images)
print(labels)