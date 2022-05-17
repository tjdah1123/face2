from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
from detectme.camera2 import FaceDetect
import time

def home(request):
    return render(request, 'home.html')

def gen(camera):
    while True:
        # prev_time = 0
        # FPS = 1
        # current_time = time.time() - prev_time
            
        # if current_time > 1./FPS:
        #     prev_time = time.time()
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@gzip.gzip_page
def detectme(request):
    return StreamingHttpResponse(gen(FaceDetect()), content_type="multipart/x-mixed-replace;boundary=frame")

def check(request):
    return render(request, "check.html")