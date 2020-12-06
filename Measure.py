# -*- coding: UTF-8 -*-

import numpy as np
import math
import sys


import cv2
import matplotlib.pyplot as plt
from skimage import measure,data,color
from PIL import Image
import glob


'''
当前假设：                                 解决办法：
1. 彩色图已经crop出目标木堆           可用深度图去除该假设
2. 拍摄的左右角度不能有较大倾斜（大于10度）  可用深度图辅助做彩色图的warp
3. 顶层发生严重弯曲
4. 非常依赖底、层两条边的正确拟合   
'''

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
    candidate[candidate < ave*1.1] = 0


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
                # for jj,val in enumerate(line_pos_temp): #剔除可能存在的outlier
                position = np.average(np.array(line_pos_temp))
                line_pos.append(position)
                line_pos_temp.clear()
                plt.axvline(x=position, ls=":", c="red")  # 平均线
    if isLine: #查出最后一条线
        position = np.average(np.array(line_pos_temp))
        line_pos.append(position)
        plt.axvline(x=position, ls=":", c="red")  # 平均线

    line_pos = np.array(line_pos)
    gaps = np.append(line_pos,0) - np.insert(line_pos,0,0)
    gaps = gaps[1:len(gaps)-2]
    gap = np.median(gaps) # 假设大部分的线是能被检测出来的


    print('统计得到的横线数量:', line_num)


    correct_num = ((line_pos[len(line_pos)-1] - line_pos[0]) / gap) + 1
    print('正确的横线数量:', correct_num)
    if line_num == np.round(correct_num):
        print('成功统计到所有横线。')
    else:
        print('未成功统计到所有横线。')

        id = 0
        while (id < line_num-1):
            if (line_pos[id+1] - line_pos[id]) > 1.5 * gap: # 间距大于1.5倍正常间距的两条线之间插入新线
                line_num +=1
                new_pos = line_pos[id] + gap
                line_pos = np.insert(line_pos, id+1, new_pos)
                plt.axvline(x=new_pos, ls=":", c="red")  # 平均线
            id +=1

    plt.savefig('./Statistics.png')
    plt.show()
    #
    # # TODO 用拟合方法得到准确的边缘位置和数量
    # pos_points = []
    # for i in range(len(line_pos)):
    #     pos_points.append((i,line_pos[i]))
    # pos_points = np.array(pos_points)
    # line = cv2.fitLine(pos_points, cv2.DIST_L2, 0, 1e-2, 1e-2)
    # k = line[1] / line[0]
    # b = line[3] - k * line[2]
    #
    # #draw
    # xx = np.linspace(0, len(line_pos), 100)
    # yy = k*xx+b
    # plt.plot(xx, yy)
    # plt.scatter(np.arange(0,len(line_pos),1), line_pos)
    # plt.show()

    # x =  int(np.ceil(-b / k))
    # pos = k*x+b
    # line_pos = []
    # line_num = 0
    # while (pos < margin.shape[1]):
    #     if candidate[pos]


    return line_num, line_pos, gap

# 直线拟合，剔除outline
def LineFitting(points, gap):
    line = cv2.fitLine(points, cv2.DIST_L2, 0, 1e-2, 1e-2)

    k = line[1] / line[0]
    b = line[3] - k * line[2]
    energy = 0
    ids = []
    max_iter = 3
    for it in range(max_iter):
        for i,point in enumerate(points): # 找内点
            dis = np.abs(k*point[0] - point[1] + b) / np.sqrt(k*k+1) #点到线的距离
            if dis < gap/3:
               ids.append(i)
               energy +=dis
        # print(len(ids), energy/len(ids))
        inlier_points = []
        for l,id in enumerate(ids):
            inlier_points.append(points[id])
        inlier_points = np.array(inlier_points)
        line = cv2.fitLine(inlier_points, cv2.DIST_L2, 0, 1e-2, 1e-2)
        k = line[1] / line[0]
        b = line[3] - k * line[2]

        energy = 0
        ids.clear()
        #TODO: 中断迭代



    return k,b


# slide a window across the image
# 分离成split列窗口
def sliding_window(image, imageRGB, pos, gap, num, split):
    window_w = int(np.floor(image.shape[1]/split))
    window_h = int(np.floor(gap/2)*2)
    for y in range(num):
        for x in range(split):
            pos_v = int(pos[y] - window_h/2) if int(pos[y] - window_h/2) >=0 else 0
            pos_u = x*window_w
            end_v = pos_v + window_h if (pos_v + window_h) < image.shape[0] else image.shape[0]

            yield (pos_v, pos_u,
                   image[ pos_v : end_v,  pos_u: pos_u+ window_w],
                   imageRGB[ pos_v : end_v,  pos_u: pos_u+ window_w])


def WidthMeasure(ks,bs,color,margin):
    # 统计直线之间的距离,更新gap
    ks = np.array(ks)
    bs = np.array(bs)

    w = color.shape[1]

    k = []
    b = []
    for i in range(len(ks)-1): #所有木板层的中间线
        k_temp = (ks[i]+ks[i+1])/2 #每条线的k为首尾k的过度
        b_temp = (bs[i]+bs[i+1])/2
        k.append(k_temp)
        b.append(b_temp)

        #画线
        p1 = (0,int(b_temp))
        p2 = (w-1, int(k_temp*(w-1)+b_temp))
        cv2.line(color, p1, p2, (255, 0, 0), 5, cv2.LINE_AA)  # draw


    # y = []
    # for i in range(len(ks)):
    #     y.append(ks[i]*w/2+bs[i]) #中间点 #
    # y = np.array(y)
    # gaps = (np.append(y, 0) - np.insert(y, 0, 0))
    # gaps = gaps[1:len(gaps) - 2]
    # gap = np.median(gaps)  # 假设所有线k=0
    #
    # layer_num = int(np.round((y[len(y)-1] - y[0]) / gap))
    # print("木板层数：", layer_num)
    # gap = int(np.round((y[len(y)-1] - y[0]) / layer_num))
    #
    # k1 = ks[0]
    # k2 = ks[len(ks)-1]
    # k = []
    # b = []
    # for i in range(layer_num): #所有木板层的中间线
    #     k_temp = k1 + (k2 - k1) * i/(layer_num-1) #每条线的k为首尾k的过度
    #     b_temp = y[0] + gap/2 + gap*i - k_temp * w/2
    #     k.append(k_temp)
    #     b.append(b_temp)
    #
    #     #画线
    #     p1 = (0,int(b_temp))
    #     p2 = (w-1, int(k_temp*(w-1)+b_temp))
    #     cv2.line(color, p1, p2, (255, 0, 0), 5, cv2.LINE_AA)  # draw

    k = np.array(k)
    b = np.array(b)

    print('木层数', len(k))

    #计算
    for x in range(w):
        y = k*x+b
        #TODO 深度图可以直接找到木板的边缘。
        # if(margin[y][x]==0):
        #     continue
        # else if
        #         asd = 1


    return




def GetMargin(gray):
    save_path = './margin.png'
    blur_size = 5  # 高斯模糊核大小
    threshold = 65  # 二值化阈值

    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0) #高斯模糊去除噪声

    # #获取边缘梯度
    # # 索比尔算子来计算x、y方向梯度
    # gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)

    # gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradY)
    #二值化
    blur = cv2.GaussianBlur(gradient, (blur_size, blur_size), 0)
    (_, margin) = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

    plt.imshow(margin,cmap='gray')
    plt.show()

    #去字
    filter_size = np.round(gray.shape[0] / 300) * 2 + 1
    img_mean = cv2.medianBlur(margin, int(filter_size))

    margin = margin - img_mean
    (_, margin) = cv2.threshold(margin, threshold, 255, cv2.THRESH_BINARY)
    cv2.imwrite(save_path, margin)

    return margin

if __name__ == '__main__':
    # img_path = './test.jpg'

    # filenames = glob.glob('E:/Projects/widthmeasure/stage2/1/*.jpg')
    # for i, img_path in enumerate (filenames):
    img_path = './samples/4.jpg'
    # img_path = './samples/test.jpg'
    color = cv2.imread(img_path, 1)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
    showImg(color, 'original color')

    # TODO: 用深度图crop出木堆区域并去除背景，得到sharp的外轮廓
    '''
    margin detection
    '''
    margin = GetMargin(gray)
    marginRGB = cv2.cvtColor(margin, cv2.COLOR_GRAY2RGB)
    showImg(margin,'mar')
    showImg(marginRGB,'margin')
    h, w = margin.shape

    # TODO：将木板旋转为正面，并使底边（或整体横线）呈水平
    Rot2Horizon()

    # TODO：判断顶层木板是否弯曲，如果不弯曲，直接可以用单条直线拟合每一条横线

    # 计算木板层数 & 得到每个条横线的位置。#TODO 现在是假设不弯曲，如果弯曲，不知道是否会让统计产生错误，之后需要用另外的sample测试下
    line_num, line_pos, gap = LayerStatistics(margin, marginRGB)


    '''
    窗口直线拟合
    '''
    # split_col = 4  #
    split_col = 1  # 窗口列数
    ks = []
    bs = []
    for (win_pos_h, win_pos_w, window, windowRGB) in sliding_window(margin, marginRGB, line_pos, gap, line_num, split_col):
        # print(win_pos_h, win_pos_w)
        contours, hierarchy = cv2.findContours(window, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
                cv2.drawContours(windowRGB, contours, i, (0, 255, 0), 2) #描边


        points = [] #属于边的点
        for i, contour in enumerate(contours):
            if i == 0 :
                points = np.reshape(contour, (-1, 2))
            points = np.concatenate((points,np.reshape(contour,(-1,2))),axis=0)

        # 直线拟合
        k,b = LineFitting(points, gap)

        # 与前一条线太近，认为是同一条，去掉。
        if len(ks)>0:
            if ( np.abs(bs[len(bs)-1]-b-win_pos_h)  < gap/2 or
                np.abs(ks[len(ks)-1]*w+bs[len(bs)-1] - (k*w+b) - win_pos_h)  < gap/2):
                continue


        ks.append(k)
        bs.append(b+win_pos_h)


        point1 = (0,int(np.round(b)))
        point2 = (window.shape[1]-1, int(np.round(k*(window.shape[1]-1)+b)))
        cv2.line(windowRGB, point1, point2, (255, 255, 255), 2, cv2.LINE_AA) #draw
        # showImg(windowRGB, 'marginrgb')

        point1_global = (win_pos_w, int(np.round(k*(win_pos_w)+b))+win_pos_h)
        point2_global = (win_pos_w + window.shape[1]-1, int(np.round(k*(win_pos_w + window.shape[1]-1)+b))+win_pos_h)
        cv2.line(color, point1_global, point2_global, (255, 255, 255), 2, cv2.LINE_AA)  # draw
        # cv2.line(marginRGB, point1_global, point2_global, (255, 0, 0), 2, cv2.LINE_AA)  # draw

    showImg(marginRGB,'margin')
    cv2.imwrite('./marginLine.png', cv2.cvtColor(marginRGB, cv2.COLOR_RGB2BGR))

    #TODO: 宽度计算 （去除缝隙等）

    WidthMeasure(ks, bs, color, margin)
    showImg(color, 'Layercolor')
    cv2.imwrite('./colorLine.png', cv2.cvtColor(color, cv2.COLOR_RGB2BGR))