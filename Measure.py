# -*- coding: UTF-8 -*-

import MarginDetection

import cv2
import matplotlib.pyplot as plt
import numpy as np


import math
import sys
from skimage import measure,data,color
from PIL import Image

def showImg(img,title):
    plt.imshow(img)
    plt.title(title)  # 图像题目
    plt.show()

# 找到底边，拟合其直线，根据直线角度将整幅图旋转到底边水平。
def Rot2Horizon():
    return


# 通过纵坐标做直方图统计，得到横边的数量、间距及粗略位置
def LayerStatistics(margin, marinRGB):
    contours, hierarchy = cv2.findContours(margin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        cv2.drawContours(marinRGB, contours, i, (0, 255, 0), 2)  # 描边

    points = []  # 属于边的点
    for i, contour in enumerate(contours):
        if i == 0:
            points = np.reshape(contour, (-1, 2))
        points = np.concatenate((points, np.reshape(contour, (-1, 2))), axis=0)

    points_v = points[:,1]
    bins_num = 200 #区间数
    data = plt.hist(points_v, bins_num, rwidth=0.5)

    bins_center = data[1] - (data[1][1] - data[1][0])/2
    bins_center = bins_center[1:len(bins_center)]    #区间中心点

    ave = len(points_v) / bins_num
    plt.axhline(y=ave, ls=":", c="red")  # 平均线


    candidate = data[0]
    candidate[candidate < ave] = 0


    #先通过直方图的峰，判断数量、位置和间距，
    # 然后通过首尾距离除以间距，验证数量。因为首尾的线大多情况比较明显，不会出现margin识别不出的问题
    isLine = False
    line_num = 0
    line_pos_temp = []
    line_pos = []
    for i, value in enumerate(candidate):
        if value > 0:
            if not isLine: #找到新线
                isLine = True
                line_num += 1
                line_pos_temp.append(bins_center[i])
            else: #用于计算新线位置
                line_pos_temp.append(bins_center[i])
        else:
            if isLine:
                isLine = False
                position = np.average(np.array(line_pos_temp))
                line_pos.append(position)
                line_pos_temp.clear()
                plt.axvline(x=position, ls=":", c="red")  # 平均线


    line_pos = np.array(line_pos)
    gaps = np.append(line_pos,0) - np.insert(line_pos,0,0)
    gaps = gaps[1:len(gaps)-2]
    gap = np.median(gaps)

    print('统计得到的横线数量、间距分别为:', line_num, gap)

    correct_num = (line_pos[len(line_pos)-1] - line_pos[0]) / gap + 1
    print('正确的横线数量:', correct_num)
    if line_num == np.round(correct_num):
        print('成功统计到所有横线。')
    else:
        print('未成功统计到所有横线。')
        a=111 # todo: 边缘检测不准，重新根据间距生成结果（数量、位置）

    plt.savefig('./Statistics.png')
    plt.show()



    return line_num, line_pos, gap

# 直线拟合，剔除outline
def LineFitting(points):
    line = cv2.fitLine(points, cv2.DIST_L2, 0, 1e-2, 1e-2)




    return line


# slide a window across the image
# 分离成split列窗口
def sliding_window(image, imageRGB, pos, gap, num, split):
    window_w = int(np.floor(image.shape[1]/split))
    window_h = int(np.floor(gap/2)*2)
    for y in range(num):
        for x in range(split):
            yield (int(pos[y] - window_h/2), x*window_w,
                   image[ int(pos[y] - window_h/2) : int(pos[y] + window_h/2), x*window_w : x*window_w+ window_w],
                   imageRGB[ int(pos[y] - window_h/2) : int(pos[y] + window_h/2), x*window_w : x*window_w+ window_w])

#small window
def sliding_window_sub(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize[1]):
        for x in range(0, image.shape[1], stepSize[0]):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])



if __name__ == '__main__':

    #read color image
    img_path = './test.jpg'
    color = cv2.imread(img_path,1)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)



    #TODO: crop出木堆区域（可通过轮廓或颜色值统计来做）

    '''
    margin detection
    '''
    margin = MarginDetection.GetMargin(img_path)
    marginRGB = cv2.cvtColor(margin, cv2.COLOR_GRAY2RGB)


    h,w = margin.shape

    #TODO：将木板旋转为底边水平
    Rot2Horizon()

    #计算木板层数 & 得到每个条横线的位置。
    line_num, line_pos, gap = LayerStatistics(margin, marginRGB)

    '''
    窗口直线拟合
    '''
    split_col = 1  # 窗口列数
    for (win_pos_h, win_pos_w, window, windowRGB) in sliding_window(margin, marginRGB, line_pos, gap, line_num, split_col):
        print(win_pos_h, win_pos_w)
        contours, hierarchy = cv2.findContours(window, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # for i, contour in enumerate(contours):
        #         cv2.drawContours(windowRGB, contours, i, (0, 255, 0), 2) #描边

        points = [] #属于边的点
        for i, contour in enumerate(contours):
            if i == 0 :
                points = np.reshape(contour, (-1, 2))
            points = np.concatenate((points,np.reshape(contour,(-1,2))),axis=0)

        # 直线拟合
        line = LineFitting(points)
        k = line[1] / line[0]
        b = line[3] - k * line[2]

        point1 = (0,int(np.round(b)))
        point2 = (window.shape[1]-1, int(np.round(k*(window.shape[1]-1)+b)))
        cv2.line(windowRGB, point1, point2, (255, 0, 0), 2, cv2.LINE_AA) #draw
        # showImg(windowRGB, 'marginrgb')
        point1_global = (win_pos_w, int(np.round(k*(win_pos_w)+b))+win_pos_h)
        point2_global = (win_pos_w + window.shape[1]-1, int(np.round(k*(win_pos_w + window.shape[1]-1)+b))+win_pos_h)
        cv2.line(color, point1_global, point2_global, (255, 0, 0), 2, cv2.LINE_AA)  # draw
        # cv2.line(marginRGB, point1_global, point2_global, (255, 0, 0), 2, cv2.LINE_AA)  # draw




    showImg(color, 'color')
    showImg(marginRGB,'margin')
    cv2.imwrite('./colorLine.png', cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    cv2.imwrite('./marginLine.png', cv2.cvtColor(marginRGB, cv2.COLOR_RGB2BGR))


    #TODO: 宽度计算 （去除缝隙等）