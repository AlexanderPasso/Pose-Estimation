# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 16:25:06 2023

@author: alexa
"""

from ultralytics import YOLO 
import cv2


model = YOLO('yolov8n-pose.pt')
cap = cv2.VideoCapture(0)

while( cap.isOpened):
    success, frame = cap.read()

    if success:
        results = model(frame, save=True)
        annotated_frame = results[0].plot()
        cv2.imshow('Yolo', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()