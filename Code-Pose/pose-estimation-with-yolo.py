# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from ultralytics import YOLO 
import cv2
import matplotlib.pyplot as plt
import numpy as np

model = YOLO('yolov8n-pose.pt')


img = cv2.imread('marcha.jpg')
cv2.imshow('Original', img)

results = model(source = img, conf=0.3)

xy = results[0].keypoints.xy

# Extrayendo puntos x
x_values = np.array(xy[0, :, 0])  # Selecciona todas las filas, segunda dimensión, primer valor en esa dimensión

# Extrayendo puntos y
y_values = np.array(xy[0, :, 1])  # Selecciona todas las filas, segunda dimensión, segundo valor en esa dimensión

# Mostrar los valores de x e y
print("Valores de x:", x_values)
print("Valores de y:", y_values)

plt.plot(x_values,y_values, '*')

