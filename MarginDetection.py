# -*- coding: UTF-8 -*-

'''
input: 木堆彩色图
function：木板边缘检测
output： 边缘二值图
'''

import cv2
import numpy as np

blur_size = 9 #高斯模糊核大小
threshold = 65 #二值化阈值

def Sobel_gradient(blurred):
    # 索比尔算子来计算x、y方向梯度
    gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)

    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    return gradient


def Thresh_and_blur(gradient):
    blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
    (_, thresh) = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    return thresh


def GetMargin(img_path):
    save_path = './margin.png'

    # 获取图片
    original_img = cv2.imread(img_path)
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)


    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0) #高斯模糊去除噪声
    gradient = Sobel_gradient(blurred) #获取边缘梯度
    margin = Thresh_and_blur(gradient) #二值化

    cv2.imwrite(save_path, margin)
    # show samples
    # cv2.namedWindow('margin', 0)
    # cv2.resizeWindow('margin', 500, 500)
    # cv2.imshow('margin', margin)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    return margin