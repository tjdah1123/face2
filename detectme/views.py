from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
from detectme.camera import FaceDetect

def home(request):
    return render(request, 'home.html')

# class VideoCamera(object):
#     def __init__(self):
#         self.video = cv2.VideoCapture(0)
#         (self.grabbed, self.frame) = self.video.read()
#         threading.Thread(target=self.update, args=()).start()

#     def get_frame(self):
#         image = self.frame
#         _, jpeg = cv2.imencode('.jpg', image)
#         return jpeg.tobytes()
    
#     def update(self):
#         while True:
#             (self.grabbed, self.frame) = self.video.read()
            
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@gzip.gzip_page
def detectme(request):
    return StreamingHttpResponse(gen(FaceDetect()), content_type="multipart/x-mixed-replace;boundary=frame")