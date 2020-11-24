'''import cv2


def get_contours(img):
    """获取连通域

    :param img: 输入图片
    :return: 最大连通域
    """
    # 灰度化, 二值化, 连通域分析
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return contours[0]


def main():
    # 1.导入图片
    img_src = cv2.imread("/home/lk/Desktop/项目/裁剪后的//thresh.png")
    img_result = img_src.copy()
    img_result=cv2.cvtColor(img_result,cv2.COLOR_BGR2GRAY)
    # 2.获取连通域
    cont = get_contours(img_src)
    cv2.drawContours(img_result, cont, -1, (0, 0, 255), 2)

    # 3.获取凸包点
    hull_points = cv2.convexHull(cont)
    cv2.polylines(img_result, [hull_points], True, (0, 255, 0), 2)

    # 4.计算 轮廓面积 与 凸包面积
    cnt_area = cv2.contourArea(cont)
    hull_area = cv2.contourArea(hull_points)
    print("轮廓面积=", cnt_area)
    print("凸包面积=", hull_area)

    # 5.计算 轮廓面积/凸包面积
    solidity = float(cnt_area) / hull_area
    print("solidity = %.4f" % solidity)

    box = cv2.boundingRect(img_result)
    cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 255, 0), 1)
    # 6.显示结果
    cv2.imshow("img_result", img_result)

    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.imwrite('/home/lk/Desktop/项目/nihe1.png',img_result)



if __name__ == '__main__':
    main()'''
# !/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math


# 处理图像，获取重心
def handle_img(src):
    l = src.shape[0]

    lx = []  # 储存X坐标
    ly = []  # 储存Y坐标
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)  # 灰度
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # 二值化
    contours, heriachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 获取轮廓

    for i, contour in enumerate(contours):
        x, y, w, h = cv.boundingRect(contour)  # 外接矩形
        mm = cv.moments(contour)  # 几何矩

        approxCurve = cv.approxPolyDP(contour, 4, True)  # 多边形逼近
        if approxCurve.shape[0] > 5:  # 多边形边大于6就显示
            cv.drawContours(src, contours, i, (0, 255, 0), 2)

        #cv.rectangle(src, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 绘制外接矩形

        # 重心
        if mm['m00'] != 0:
            cx = mm['m10'] / mm['m00']
            cy = mm['m01'] / mm['m00']
            cv.circle(src, (np.int(cx), np.int(cy)), 3, (0, 0, 255), -1)  # 绘制重心
            lx.append(np.int(cx))  # 翻转x坐标，图片坐标系原点不在下边
            ly.append(l - np.int(cy))

    cv.imshow("handle_img", src)
    cv.imwrite("/home/lk/Desktop/项目/裁剪后的/123thresh6.png", src)
    return lx, ly


# 处理坐标点，分组
def handle_point(x, y):
    # 排序
    lx = []
    ly = []
    rx = []
    ry = []
    points = zip(x, y) #获取点
    sorted_points = sorted(points)
    x = [point[0] for point in sorted_points]
    y = [point[1] for point in sorted_points]

    # 分割
    Max = 0
    for i in range(len(x) - 1):      #找到左右两边点最大间隔
        d = np.int(math.hypot(x[i + 1] - x[i], y[i + 1] - y[i]))
        if d > Max:
            Max = d
            k = i
    for i in range(len(x)):   #区分左右点
        if i < k + 1:
            lx.append(x[i])
            ly.append(y[i])
        else:
            rx.append(x[i])
            ry.append(y[i])
    return lx, ly, rx, ry


# 拟合，画图
def poly_fitting(lx, ly, rx, ry):
    lx = np.array(lx)
    ly = np.array(ly)
    rx = np.array(rx)
    ry = np.array(ry)

    fl = np.polyfit(lx, ly, 3)  # 用3次多项式拟合
    pl = np.poly1d(fl)  # 求3次多项式表达式
    print("左边：", pl)
    lyy = pl(lx)  # 拟合y值

    fr = np.polyfit(rx, ry, 3)  # 用3次多项式拟合
    pr = np.poly1d(fr)  # 求3次多项式表达式
    print("右边：", pr)
    ryy = pr(rx)  # 拟合y值

    # 绘图
    plot1 = plt.plot(lx, ly, 'r*')
    plot2 = plt.plot(lx, lyy, 'b')
    plot3 = plt.plot(rx, ry, 'r*')
    plot4 = plt.plot(rx, ryy, 'b')
    plt.title('poly_fitting')
    plt.show()


if __name__ == "__main__":
    img = cv.imread("/home/lk/Desktop/项目/裁剪后的/123thresh.png")
    cv.imshow("src", img)

    x, y = handle_img(img)
    #lx, ly, rx, ry = handle_point(x, y)
    #poly_fitting(lx, ly, rx, ry)

    cv.waitKey(0)
    cv.destroyAllWindows()


