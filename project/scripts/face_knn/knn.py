#coding=utf-8
# import cv2
# import face_recognition

# import math
# from sklearn import neighbors
# import os
# import os.path
# from PIL import Image, ImageDraw
# from face_recognition.face_recognition_cli import image_files_in_folder
from knnclass import *


if __name__ == "__main__":
    print "Start face recognition program"
    zhang = face_classification()
    # zhang.train("picture/know")
    # a=zhang.predict("zhangzeyong.JPG",False)
    # print a
    b = zhang.video_predict("zhangzeyong",True)
    print b

