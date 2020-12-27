# -*- coding: UTF-8 -*-

import numpy as np
import math
import sys
import glob
import time
import cv2
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema



# import pyransac3d as ransac3d

# from skimage import measure,data,color
# from PIL import Image

import AutoBrightness as aug
import Geometry as geo

'''
当前假设：                           
3. 顶层未发生严重弯曲
4. 非常依赖底、顶层两条边的正确拟合   
'''



def showImg(img,title):
    plt.imshow(img)
    plt.title(title)  # 图像题目
    plt.show()

def showImgGray(img,title):
    plt.imshow(img, cmap='gray')
    plt.title(title)  # 图像题目
    plt.show()


# 通过纵坐标做直方图统计，得到横边的数量、间距及粗略位置
# todo 统计会出现很多边连在一起只被识别未一个
def LayerStatistics(margin, marinRGB):
    '''通过纵坐标做直方图统计，得到横边的数量、间距及粗略位置
    获取所有属于边缘的点的纵坐标,然后将纵坐标划分为100个区间,然后在直方图上表示出每个区间的点数,并标记出区域中心点,
    将所有点除以区间获得一个均值,设置一个inlier点的概率,与均值乘积并设置一个平行于x轴的直线表示出来,做一个筛选,将
    点数少于这个值的区间清空,然后通过直方图的峰，判断数量、位置和间距，然后通过首尾距离除以间距，验证数量,正确数目为
    最后一条线的位置减去第一条除以间距加上一,如果数目对,就结束,如果不对,采用设置的距离阈值乘以距离,如果木板间距大于
    这个值,就在之间插入新的线然后再统计数目。


    :param margin:木板图二值化后的图
    :param marinRGB:原图
    :return:线的数目,位置以及间距
    '''
    bins_num = 100  # 区间数
    inlier_bin = 0.8

    contours, hierarchy = cv2.findContours(margin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        cv2.drawContours(marinRGB, contours, i, (0, 255, 0), 2)  # 描边

    points = []  # 属于边的点
    for i, contour in enumerate(contours):
        if i == 0:
            points = np.reshape(contour, (-1, 2))
        points = np.concatenate((points, np.reshape(contour, (-1, 2))), axis=0)

    points_v = points[:,1]

    data = plt.hist(points_v, bins_num, rwidth=0.5)

    bins_center = data[1] - (data[1][1] - data[1][0])/2
    bins_center = bins_center[1:len(bins_center)]    #区间中心点

    ave = len(points_v) / bins_num
    plt.axhline(y=ave*inlier_bin, ls=":", c="red")  # 平均线


    candidate = data[0]
    candidate[candidate < ave*inlier_bin] = 0


    #先通过直方图的峰，判断数量、位置和间距，
    # 然后通过首尾距离除以间距，验证数量。因为首尾的线大多情况比较明显，不会出现margin识别不出的问题
    isLine = False
    line_num = 0
    line_pos_temp = []
    line_pos = []
    line_ids_temp = []
    line_ids = []
    #找属于线的bin
    for i, value in enumerate(candidate):
        if value > 0:
            if not isLine: #找到新线
                isLine = True
                line_num += 1
                line_pos_temp.append(bins_center[i])
                line_ids_temp.append(i)
            else: #用于计算新线位置
                line_pos_temp.append(bins_center[i])
                line_ids_temp.append(i)
        else:
            if isLine:
                isLine = False
                # for jj,val in enumerate(line_pos_temp): #剔除可能存在的outlier
                position = line_pos_temp[0]
                line_ids.append(line_ids_temp[0])
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


    print('统计得到的木层边数:', line_num)

    #添加边的阈值#todo 添加未检测到的底边（可用深度图直接找到）
    gap_thre = 1.3
    correct_num = ((line_pos[len(line_pos)-1] - line_pos[0]) / gap) + 1
    print('正确的横线数量:', correct_num)
    if line_num == np.round(correct_num):
        print('成功统计到所有边数。')
    else:
        print('未成功统计到所有边数,进行重采样')

        id = 0
        while (id < line_num-1):
            if (line_pos[id+1] - line_pos[id]) > gap_thre * gap: # 间距大于1.5倍正常间距的两条线之间插入新线
                line_num +=1
                new_pos = line_pos[id] + gap
                line_pos = np.insert(line_pos, id+1, new_pos)
                plt.axvline(x=new_pos, ls=":", c="red")  # 平均线
            id +=1
    print('重采样结果:', line_num)
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


# slide a window across the image
# 分离成split列窗口
def sliding_window(image, imageRGB, pos, gap, num, split):
    '''
        将整幅图划分为一个个窗口,可根据前一步获取的边缘直线确定窗口位置,防止出现窗口中没有或出现多条边缘直线的情况


        :param image:二值化的图
        :param imageRGB:RGB图
        :param pos:边缘直线位置
        :param gap:边缘直线距离
        :param num:直线数目
        :param split:窗口列数
        :return:
    '''
    window_w = int(np.floor(image.shape[1]/split))
    window_h = int(np.floor(gap/2)*3)


    for y in range(num):
        for x in range(split):
            pos_v = int(pos[y] - window_h/2) if int(pos[y] - window_h/2) >=0 else 0
            pos_u = x*window_w
            end_v = pos_v + window_h if (pos_v + window_h) < image.shape[0] else image.shape[0]

            yield (pos_v, pos_u,
                   image[ pos_v : end_v,  pos_u: pos_u+ window_w],
                   imageRGB[ pos_v : end_v,  pos_u: pos_u+ window_w])

#通过遍历做直线拟合
def sliding_window_easy(image, imageRGB, pos, gap, num, split):

    window_w = int(np.floor(image.shape[1]/split))
    window_h = int(np.floor(gap/2*2))
    step = int(window_h/2)
    win_num_h = int(np.floor(image.shape[0]/step))+1 #移动步长=半个gap

    for y in range(win_num_h):
        for x in range(split):
            pos_v = int(y*step)
            pos_u = x*window_w
            end_v = pos_v + window_h if pos_v + window_h < image.shape[0] else image.shape[0]-1
            yield (pos_v, pos_u,
                   image[ pos_v : end_v,  pos_u: pos_u+ window_w],
                   imageRGB[ pos_v : end_v,  pos_u: pos_u+ window_w])




def LREdgeDetection(gray, candidate):
    edgepoint = []
    u_bound = int(gray.shape[1] / 30)
    v_bound = int(gray.shape[0] / 50)
    u1 = candidate[1] - u_bound if candidate[1] - u_bound >= 0 else 0
    u2 = candidate[1] + u_bound if candidate[1] + u_bound <= gray.shape[1] - 1 else gray.shape[1] - 1
    v1 = candidate[0] - v_bound if candidate[0] - v_bound >= 0 else 0
    v2 = candidate[0] + v_bound if candidate[0] + v_bound <= gray.shape[0] - 1 else gray.shape[0] - 1

    patch = gray[v1:v2, u1:u2].copy()
    showImgGray(patch, 'patch')
    # #获取边缘梯度
    threshold = 50  # 二值化阈值
    gradX = cv2.Sobel(patch, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    gradient1 = cv2.convertScaleAbs(gradX)
    (_, margin1) = cv2.threshold(gradient1, threshold, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(margin1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    points = []  # 属于边的点
    for i, contour in enumerate(contours):
        if i == 0:
            points = np.reshape(contour, (-1, 2))
        points = np.concatenate((points, np.reshape(contour, (-1, 2))), axis=0)


    bins_num = 10
    points_u = points[:, 0]

    data = plt.hist(points_u, bins_num, rwidth=0.5)

    bins_center = data[1] - (data[1][1] - data[1][0]) / 2
    bins_center = bins_center[1:len(bins_center)]  # 区间中心点
    id_max = np.argmax(data[0])



    plt.imshow(margin1, cmap='gray')
    plt.title('edgedetect')  # 图像题目
    plt.axvline(x=bins_center[id_max], ls=":", c="red")  # 平均线
    plt.show()

    if data[0][id_max] < 30: #TODO
        return candidate
    else:
        return [candidate[0], int(bins_center[id_max] + u1)]




def WidthMeasure(ks,bs,gray,color,depth,z):
    '''
        找到每层木板的中间线,依靠灰度检测缝隙并用白线画出来,返回每层木板的宽度
        #TODO 目前只采用depth做检测
        :param ks:直线k集合
        :param bs: 直线b集合
        :param gray: 灰度图
        :param color: 彩色图
        :return:每层木板宽度
    '''
    # 统计直线之间的距离,更新gap
    ks = np.array(ks)
    bs = np.array(bs)
    vdepth = depth.copy()
    w = gray.shape[1]

    #算木板中线
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
        # cv2.line(color, p1, p2, (255, 0, 0), 3, cv2.LINE_AA)  # draw

    k = np.array(k)
    b = np.array(b)

    print('最终木层数', len(k))

    # 检测左右边
    bound = []
    for i in range(len(k)):
        leftedge = []
        rightedge = []
        # 用一条线检测
        for x in range(int(w)):
            y = int(np.round(k[i] * x + b[i]))
            if depth[y,x]:
                l = [y,x]
                # check by gray
                leftedge = l
                if i == len(k)-1:
                    leftedge = LREdgeDetection(gray, l) #只有最底下的边会被优化，因为彩色图边缘不明显
                break
        for x in range(int(w)):
            y = int(np.round(k[i] * (w-1-x) + b[i]))
            if depth[y,w-1-x]:
                r = [y,w-1-x]
                rightedge = r
                if i == len(k) - 1:
                    rightedge = LREdgeDetection(gray, r)
                break
        bound.append([leftedge,rightedge])

        offset = 0        # trick TODO 左边偏少右边偏多
        cv2.line(color, (leftedge[1]-offset,leftedge[0]), (rightedge[1]-offset,rightedge[0]), (255, 0, 0), 4, cv2.LINE_AA)  # draw
        cv2.line(vdepth, (leftedge[1] - offset, leftedge[0]), (rightedge[1] - offset, rightedge[0]), 128, 4, cv2.LINE_AA)  # draw
    bound = np.array(bound)


    #-----------------检测缝隙--------------------
    #统计所有中线路径上的颜色
    #认为缝隙的颜色小于一个阈值（黑色）
    #彩色图上检测到的大缝隙需要深度图也检测到（排除大的黑块对结果的影响）
    #depth图辅助将彩图没检测到的检测出来
    pt_values = []
    for i in range(len(k)):
        for x in range(bound[i, 0, 1], bound[i, 1, 1], 1):
            y = int(np.round(k[i] * x + b[i]))
            pt_values.append(gray[y,x])

    pt_values = np.array(pt_values)

    bins_num = int(256/4)
    data = plt.hist(pt_values, bins_num, rwidth=0.5)

    bins_center = data[1] - (data[1][1] - data[1][0]) / 2
    bins_center = bins_center[1:len(bins_center)]  # 区间中心点

    # ave = len(points_v) / bins_num
    # plt.axhline(y=ave * inlier_bin, ls=":", c="red")  # 平均线
    plt.show()

    mi, ma = argrelextrema(data[0], np.less)[0], argrelextrema(data[0], np.greater)[0]
    cutoff = bins_center[mi[2]]
    print(cutoff)


    thre_color = cutoff
    pts = []
    thre_min_dis = int(np.floor(gray.shape[1] / 300))  # 缝隙不小于3000/300个像素
    thre_wid = int(thre_min_dis *2)
    thre_min_num_ratio = 0.4  # 缝隙3*2宽度内像素数不小于thre_min_num_ratio*面积个像素
    thre_max_check = thre_min_dis * 10  # 缝隙很大时需用深度图验证
    for i in range(len(k)):
        pt1 = []
        pt2 = []
        find = False
        ptline = []
        for x in range(bound[i,0,1], bound[i,1,1],1):
            y = int(np.round(k[i] * x + b[i]))
            if not find:
                if gray[y,x] <= thre_color:
                    find = True
                    pt1 = [y,x]
            else:
                if gray[y,x] > thre_color:
                    pt2 = [y,x-1]
                    find = False

                    if pt2[1] - pt1[1] >= thre_min_dis: # 满足最小缝隙阈值

                        if pt2[1] - pt1[1] >= thre_max_check:
                            patch = depth[pt1[0] - thre_wid: pt1[0] + thre_wid + 1, pt1[1]: pt2[1] + 1]
                            pt_inliers = np.where(patch == 0)[0]
                            if len(pt_inliers) < patch.shape[0] * patch.shape[1] * thre_min_num_ratio:
                                continue

                        patch = gray[pt1[0]-thre_wid : pt1[0]+thre_wid+1, pt1[1] : pt2[1]+1]
                        pt_inliers = np.where(patch < cutoff)[0]
                        if len(pt_inliers) >= patch.shape[0]*patch.shape[1]*thre_min_num_ratio:
                            ptline.append([pt1,pt2])
                            for j in range(len(pts)):  # 画缝隙
                                cv2.line(color, (pt1[1],pt1[0]), (pt2[1],pt2[0]), (0, 0, 0), 4, cv2.LINE_AA)  # draw
                                cv2.line(vdepth, (pt1[1], pt1[0]), (pt2[1], pt2[0]), 0, 4,
                                         cv2.LINE_AA)  # draw

        pts.append(ptline)
    pts = np.array(pts)
    showImgGray(vdepth,'d')
    # #只用彩色图
    # pts = []
    # thre = 40 # gray value小于thre就是缝隙
    # for i in range(len(k)):
    #     pt1 = []
    #     pt2 = []
    #     find = False
    #     for x in range(int(w - w/10)):
    #         if x < w/10:
    #             continue
    #         y = int(k[i]*x+b[i])
    #         # TODO 深度图可以直接找到木板的边缘。
    #         if not find:
    #             if gray[y][x]<thre:
    #                 find = True
    #                 pt1 = (x,y)
    #         else:
    #             if gray[y][x]>thre:
    #                 find = False
    #                 pt2 = (x,y)
    #                 if pt2[0]-pt1[0]>5:
    #                     pts.append(pt1)
    #                     pts.append(pt2)
    # for i in range(int(len(pts)/2)): #画缝隙
    #     cv2.line(color, pts[i*2], pts[i*2+1], (255, 255, 255), 2, cv2.LINE_AA)  # draw


    wids = []
    for id in range(int(len(k))):
        focal = geo.intrinsic[0]
        delta = abs(bound[id,1] - bound[id,0]) + [0,1] # 长度+1,#todo
        length = np.sqrt(delta[0] ** 2 + delta[1] ** 2) / focal * z

        for id1 in range(int(len(pts[id]))):
            delta = abs(np.array(pts[id][id1][1]) - np.array(pts[id][id1][0])) + [0,1]
            length_margin = np.sqrt(delta[0] ** 2 + delta[1] ** 2) / focal * z
            length -=length_margin
        wids.append(length)



    wids = np.array(wids)
    return wids




def GetMargin(gray):
    '''
        得到二值化的图,并对木板中的字之类的集中轮廓进行处理

        :param gray:灰度图
        :return:处理后的二值化图
    '''
    save_path = './margin.png'
    blur_size = 3  # 高斯模糊核大小
    threshold = 50  # 二值化阈值

    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0) #高斯模糊去除噪声

    # #获取边缘梯度
    # # 索比尔算子来计算x、y方向梯度
    gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    gradient1 = cv2.convertScaleAbs(gradX)
    gradient2 = cv2.convertScaleAbs(gradY)
    (_, margin1) = cv2.threshold(gradient1, threshold, 255, cv2.THRESH_BINARY)
    (_, margin2) = cv2.threshold(gradient2, threshold, 255, cv2.THRESH_BINARY)

    #异或
    # bitwiseXor = cv2.bitwise_xor(margin1,margin2 )
    # showImgGray(bitwiseXor,'bitwiseXor')

    # element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    # dilate = cv2.dilate(bitwiseXor,element)
    # erode = cv2.erode(dilate,element)

    # showImgGray(erode, 'erode')

    #去字等等集中轮廓
    filter_size = np.round(gray.shape[0] / 200) * 2 + 1
    img_mean = cv2.medianBlur(margin2, int(filter_size))
    margin2 = margin2 - img_mean
    showImgGray(margin2,'margindelete')

    min_line_length = gray.shape[1]/5
    max_line_gap = min_line_length/20
    thres = int(min_line_length/2)


    lines = cv2.HoughLinesP(margin2, 2, np.pi / 180, thres, min_line_length, max_line_gap)
    if lines.shape[0] == 0:
        print("HoughLinesP get no lines!")
        exit()
    thre_min_len = gray.shape[1] / 100
    map = np.zeros(gray.shape)
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(x2 - x1) == 0:
                continue
            else:
                if abs(y2-y1)/abs(x2-x1) > 0.2 or abs(y2-y1)+abs(x2-x1) < thre_min_len:
                    continue

            cv2.line(map, (x1, y1), (x2, y2), 255, 3)

    margin = cv2.convertScaleAbs(map)
    showImgGray(margin, 'map')

    cv2.imwrite(save_path, margin)

    return margin

if __name__ == '__main__':
    # filenames = glob.glob('E:/Projects/widthmeasure/stage2/1/*.jpg')
    # for i, img_path in enumerate (filenames):
    tstart = time.time()
    color_img_path = './4k/colorFrame1609044002.png'
    depth_img_path = './4k/tranColorFrame1609044002.png'

    color = cv2.imread(color_img_path, cv2.IMREAD_ANYCOLOR)
    depth = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)

    if not len(color) or not len(depth):
        print("Read color failed!")
        exit()

    #亮度增强
    color = aug.AutoBright(color)

    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    showImg(color, 'original color')

    # showImg(depth, 'original depth')
    depth[depth > 1500] = 0
    depth[depth < 500] = 0
    showImg(depth, 'original depth')



    #'''平面拟合'''
    time1 = time.time()
    pointcloud = geo.BackProjection(depth)

    time2 = time.time()
    print(f"back-projection time: {time2 - time1}")

    #TODO 底边会连着地面
    plane_model, inlier_cloud = geo.PlaneFitting(pointcloud, 0.05, 10, 30, 0.01, 20)

    time3 = time.time()
    print(f"plane fitting time: {time3 - time2}")

    #DEBUG
    # color_debug = color.copy()
    # uv = geo.To2D(inlier_cloud)
    # for i,value in enumerate(uv):
    #     u = int(np.round(value[0]))
    #     v = int(np.round(value[1]))
    #     color_debug[v,u,:] = color_debug[v,u,:]/2

    time4 = time.time()
    # print(f"projection time: {time4 - time3}")



    #Crop + Warp 得到平行于图像与底边的目标木堆平面图
    plane_color, plane_depth, plane_z = geo.GetWarpedImg(plane_model, inlier_cloud, color)

    time5 = time.time()
    print(f"Warpping time: {time5 - time4}")

    cv2.imwrite('./planeColor5.png', cv2.cvtColor(plane_color, cv2.COLOR_RGB2BGR))
    cv2.imwrite('./planeDepth5.png', plane_depth)
    print(plane_z)
    color = plane_color

    # color_img_path = './planeColor5.png'
    # depth_img_path = './planeDepth5.png'
    # color  = cv2.imread(color_img_path, cv2.IMREAD_ANYCOLOR)
    # plane_depth = cv2.imread(depth_img_path, cv2.IMREAD_GRAYSCALE)
    # plane_z = 1.633
    #
    # color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    # plane_color = color
    # showImg(plane_color, 'original color')
    # showImgGray(plane_depth, 'original depth')

    ##'''窗口直线拟合'''----------------------------------------------------------------------------------
    gray = cv2.cvtColor(plane_color, cv2.COLOR_RGB2GRAY)
    #margin detection
    margin = GetMargin(gray)
    marginRGB = cv2.cvtColor(margin, cv2.COLOR_GRAY2RGB)
    # showImg(marginRGB,'margin')
    h, w = margin.shape


    # TODO：判断顶层木板是否弯曲，如果不弯曲，直接可以用单条直线拟合每一条横线,
    #  现在是假设不弯曲，如果弯曲，不知道是否会让统计产生错误，之后需要用另外的sample测试下

    # 计算木板层数 & 得到每个条横线的位置。
    line_num, line_pos, gap = LayerStatistics(margin, marginRGB)


    '''
    窗口直线拟合
    '''
    # split_col = 4  #
    split_col = 1  # 窗口列数
    ks = []
    bs = []
    inlier_nums = []
    #todo 更优的位置查找，和直线拟合
    for (win_pos_h, win_pos_w, window, windowRGB) in sliding_window(margin, marginRGB, line_pos, gap, line_num, split_col):
    # for (win_pos_h, win_pos_w, window, windowRGB) in sliding_window_easy(margin, marginRGB, line_pos, gap, line_num, split_col):
        # print(win_pos_h, win_pos_w)
        contours, hierarchy = cv2.findContours(window, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
                cv2.drawContours(windowRGB, contours, i, (0, 255, 0), 2) #描边


        points = [] #属于边的点
        for i, contour in enumerate(contours):
            if i == 0 :
                points = np.reshape(contour, (-1, 2))
            points = np.concatenate((points,np.reshape(contour,(-1,2))),axis=0)
        if len(points) < 30:
            continue
        # 直线拟合
        k,b,inlier_num = geo.LineFitting(points, gap)


        # todo 与前一条线太近，认为是同一条，去掉。
        if len(ks)>0:
            print(math.degrees(np.abs(math.atan(ks[len(ks) - 1]) - math.atan(k))))
            if  np.abs(math.atan(ks[len(ks)-1])-math.atan(k)) > np.pi/100: # 与上一条斜率差太多（5度）就重置斜率
                midpx = window.shape[1]/2
                midpy = k*midpx+b
                k = ((ks[len(ks)-1]) *3 + k)/4 #更相信上一条线的斜率
                b = midpy-k*midpx
                # continue
            if ( np.abs(bs[len(bs)-1]-b-win_pos_h)  < gap/3 or
                np.abs(ks[len(ks)-1]*w+bs[len(bs)-1] - (k*w+b+ win_pos_h))  < gap/3):
                if inlier_nums[len(ks)-1] *3 < inlier_num: #删除上一条线
                    ks.pop()
                    bs.pop()
                    inlier_nums.pop()
                else: #当前为重复的线，剔除
                    continue




        ks.append(k)
        bs.append(b+win_pos_h)
        inlier_nums.append(inlier_num)

        '''
        窗口内直线的点用白色线连接
        '''
        point1 = (0,int(np.round(b)))
        point2 = (window.shape[1]-1, int(np.round(k*(window.shape[1]-1)+b)))
        cv2.line(windowRGB, point1, point2, (255, 255, 255), 1, cv2.LINE_AA) #draw
        # showImg(windowRGB, 'windowrgb')
        '''
        找到目标点在整幅图上的位置,并用白线连接
        '''
        point1_global = (win_pos_w, int(np.round(k*(win_pos_w)+b))+win_pos_h)
        point2_global = (win_pos_w + window.shape[1]-1, int(np.round(k*(win_pos_w + window.shape[1]-1)+b))+win_pos_h)
        cv2.line(color, point1_global, point2_global, (255, 255, 255), 1, cv2.LINE_AA)  # draw
        # cv2.line(marginRGB, point1_global, point2_global, (255, 0, 0), 2, cv2.LINE_AA)  # draw

    showImg(marginRGB,'margin')
    cv2.imwrite('./marginLine.png', cv2.cvtColor(marginRGB, cv2.COLOR_RGB2BGR))

    #TODO: 宽度计算 （去除缝隙等）
    wids = WidthMeasure(ks, bs, gray, color, plane_depth, plane_z)



    for i in range(len(wids)):
        print(f"{wids[i] * 100.0:.2f}")

    tend = time.time()
    print(f"total cost time(s): {tend-tstart}")
    showImg(color, 'Layercolor')
    cv2.imwrite('./colorLine.png', cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
