import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from skimage import measure,data,color
from PIL import Image

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize[1]):
        for x in range(0, image.shape[1], stepSize[0]):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


# 返回滑动窗结果集合，本示例暂时未用到
def get_slice(image, stepSize, windowSize):
    slice_sets = []
    for (x, y, window) in sliding_window(image, stepSize, windowSize):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != windowSize[1] or window.shape[1] != windowSize[0]:
            continue
        slice = image[y:y + windowSize[1], x:x + windowSize[0]]
        slice_sets.append(slice)
        print(slice_sets)
    return slice_sets

def handle_img(slice):
    l = slice.shape[0]

    lx = []  # 储存X坐标
    ly = []  # 储存Y坐标
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 灰度
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 二值化
    contours, heriachy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 获取轮廓

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)  # 外接矩形
        mm = cv2.moments(contour)  # 几何矩

        approxCurve = cv2.approxPolyDP(contour, 4, True)  # 多边形逼近
        if approxCurve.shape[0] > 2:  # 多边形边大于6就显示
            cv2.drawContours(image, contours, i, (0, 255, 0), 2)

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 绘制外接矩形

        # 重心
        if mm['m00'] != 0:
            cx = mm['m10'] / mm['m00']
            cy = mm['m01'] / mm['m00']
            cv2.circle(image, (np.int(cx), np.int(cy)), 3, (0, 0, 255), -1)  # 绘制重心
            lx.append(np.int(cx))  # 翻转x坐标，图片坐标系原点不在下边
            ly.append(l - np.int(cy))


    return lx, ly

def duobianxingbijin():
    img = cv2.imread('/home/lk/Desktop/项目/裁剪后的/thresh2.png')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)  # 外接矩形
        mm = cv2.moments(contour)  # 几何矩

        approxCurve = cv2.approxPolyDP(contour, 0, True)  # 多边形逼近
        if approxCurve.shape[0] > 3:  # 多边形边大于6就显示
            cv2.drawContours(img, contours, i, (0, 255, 0), 2)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    #cv2.imwrite('/home/lk/Desktop/项目/裁剪后的/thresh21.png', img)
    #cv2.imshow("img", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return img


def green_points(slice):
    #slice=cv2.imread('/home/lk/Desktop/项目/裁剪后的/finaladf (copy).png')
    print(slice.shape)

    h=slice.shape[0]
    w=slice.shape[1]
    c=slice.shape[2]
    green_pix_ind=[0,255,0]
    red_pix_ind=[0,0,255]
    print('weight : % s,height : % s'%(w,h))
    a = []
    b = []
    d = []
    for row in range(h):
        for col in range(w):
            for channel in range(c):
                if( (slice[(row,col)][:]==green_pix_ind ).all() ):
                    pv=(row,col)

                    a.append(row)
                    b.append(col)
                    d.append(pv)
                    #print(a)
                    #slice[(row, col)][:] = red_pix_ind
    #cv2.imshow('sdf',slice)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #a.reverse(a)
    return a,b,d
def bianlislice(slice):
    d1=[]
    h1 = slice.shape[0]
    w1 = slice.shape[1]
    for row in range(h1):
        for col in range(w1):
            pv1=(row,col)
            d1.append(pv1)
def jinghua(slice):
    print(slice.shape)
    a=[]
    h=slice.shape[0]
    w=slice.shape[1]
    channels=slice.shape[2]
    print('weight : % s,height : % s'%(w,h))
    for row in range(h):
        for col in range(w):
            for c in range(channels):

                pv=slice[row,col,c]
                while(pv==255):

                    a.append(pv)
    for slice in slice_sets:
        while(a.len<=w):
            a[:]=0

    return pv
'''def nihe(slice):
    a=[]
    global  lx
    global ly
    points = zip(x, y)  # 获取点
    sorted_points = sorted(points)
    u = [point[0] for point in sorted_points]
    v = [point[1] for point in sorted_points]

    return lx,ly'''
def xianshi(lx,ly):
    lx = np.array(x)
    ly = np.array(y)
    fl = np.polyfit(lx, ly, 2)
    plot = plt.plot(lx, ly, 'r*')
    plt.title('直线拟合')
    plt.show()




def handle_point(x, y):
    # 排序
    ux = []
    uy = []
    vx = []
    vy = []
    points = zip(x, y) #获取点
    sorted_points = sorted(points)
    x = [point[0] for point in sorted_points]
    y = [point[1] for point in sorted_points]

    # 分割
    '''Max = 0
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
    return lx, ly, rx, ry'''


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





def Sobel_gradient(blurred):
    # 索比尔算子来计算x、y方向梯度
    gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)

    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    return gradX, gradY, gradient
#def nihe(point):

def linear_regression(x, y):
  N = len(x)
  sumx = sum(x)
  sumy = sum(y)
  sumx2 = sum(x ** 2)
  sumxy = sum(x * y)
  A = np.mat([[N, sumx], [sumx, sumx2]])
  b = np.array([sumy, sumxy])
  return np.linalg.solve(A, b)



def Least_squares(x,y):
    x_ = sum(x)/len(x)
    y_ = sum(y)/len(y)
    m = np.zeros(1)
    n = np.zeros(1)
    k = np.zeros(1)
    p = np.zeros(1)
    for i in np.arange(50):
        k = (x[i]-x_)* (y[i]-y_)
        m += k
        p = np.square( x[i]-x_ )
        n = n + p
    a = m/n
    b = y_ - a* x_
    return a,b



def tichufengxi(slice2):    #剔除缝隙像素函数
    w,h,c=slice2.shape
    heng=[]
    zong=[]
    zuobiao=[]
    for row in range(w):
        for col in range(h):
            if ( (slice2[(row,col)][:]==[0,0,0] ).all() ):
                heng.append(row)
                zong.append(col)
                pv=(row,col)
                zuobiao.append(pv)
    return heng,zong,zuobiao



















if __name__ == '__main__':
    #image = cv2.imread('/home/lk/Desktop/项目/裁剪后的/thresh.png')
    img = cv2.imread('/home/lk/Desktop/项目/裁剪后的/thresh2.png')
    img1 = cv2.imread('/home/lk/Desktop/项目/dd61ef3905306ad70e8fe659f52c271.jpg')
    #img = cv2.imread(sys.argv[1], cv2.CV_LOAD_IMAGE_UNCHANGED)
    b = img1[:, :, 0]
    g = img1[:, :, 1]
    r = img1[:, :, 2]
    #cv2.imshow('img', img)
    RGB=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)          #由于opencv的问题颜色需要转换
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)  # 外接矩形
        mm = cv2.moments(contour)  # 几何矩

        approxCurve = cv2.approxPolyDP(contour, 0, True)  # 多边形逼近
        if approxCurve.shape[0] > 3:  # 显示多边形
            cv2.drawContours(img, contours, i, (0, 255, 0), 2) #描边
    image=img
    # 自定义滑动窗口的大小
    w = image.shape[1]
    h = image.shape[0]

    (winW, winH) = (int(w/6),int(h/22.5))     #窗口大小
    stepSize = (int (w/6),int(h/22.5))
    cnt = 0
    zhixian = []
    pianyi=[]
    houdu=[]
    changdu=[]
    c = 1
    heng1=[]
    zong1=[]
    zuobiao1=[]




    for (x, y, window) in sliding_window(image, stepSize=stepSize, windowSize=(winW, winH)): #每一个窗口

        if window.shape[0] != winH or window.shape[1] != winW:  #窗口不符合设定大小就抛弃
            continue

        clone = image.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 0, 255), 2)
        cv2.imshow("Window", clone)


        slice = image[y:y+winH,x:x+winW]         #设置不同图片的切片,有描边后的阈值图,还有原图,为了之后效果显示在原图上
        slice1=RGB[y:y+winH,x:x+winW]
        slice2=img1[y:y+winH,x:x+winW]
        cv2.namedWindow('sliding_slice1', 0)
        cv2.imshow('sliding_slice1', slice2)
        slice_sets1 = []
        slice_sets1.append(slice1)

        cv2.namedWindow('sliding_slice',0)
        cv2.imshow('sliding_slice', slice)
        slice_sets = []                       #所有切片集合
        slice_sets.append(slice)

        slice_sets2=[]
        slice_sets2.append(slice2)

        #zhixian=np.empty(shape=[4,20],dtype=float)

        count = 0
        diyige = []                #第一个和最后一个的序号集合,为了后面识别窗口位置
        zuihouyige = []
        a1 = 1
        b1 = 6

        for count1 in range(20):          #因为横向切了四片所以+4
            diyige.append(a1)
            a1 = a1 + 6

        for count2 in range(20):
            zuihouyige.append(b1)
            b1 = b1 + 6
        #print(diyige)
        #('this is slice_sets: ', len(slice_sets.))
        #cv2.imshow('sdfsd',img1)


        for i in range(1):


            heng, zong, zuobiao = tichufengxi(slice2)

            heng1.append(heng)
            zong1.append(zong)
            zuobiao1.append(zuobiao)
            print(heng1[c - 1])

            a,b,d=green_points(slice)
            '''q,w,d5=green_points(slice)#获取slice中所有绿色点
            a=[row for row in q if row not in heng] #去除木板间缝隙点
            b = [col for col in w if col not in zong]
            d = [pv for pv in d5 if pv not in zuobiao]'''
            d1=bianlislice(slice)
            v = a
            u = b
            p = []
            a1, b1, d1 = green_points(slice)
            v1 = a1
            u1 = b1
            '''for i in range(len(d)):
                p.append((d[i][1], d1[i][0]))
            sorted_point = sorted(p)
            u1 = [point[0] for point in sorted_point]
            v1 = [point[1] for point in sorted_point]'''

            if len(v)!=0:                  #窗口可能会没有绿色像素点,绿色为描边颜色,有的话才继续
                z = max(v)                            #对绿色点进行预处理,去掉干扰点
                q = min(v)
            else:
                continue
            length = z - q
            yi = []
            er = []
            san = []
            si = []
            wu = []
            liu = []
            qi = []
            ba = []
            yi1 = []
            er1 = []
            san1 = []
            si1 = []
            wu1 = []
            liu1 = []
            qi1 = []
            ba1 = []
            jiu1 = []
            shi1 = []
            shiyi1 = []
            shier1 = []
            shisan1 = []
            shisi1 = []
            shiwu1 = []
            e = max(u1)
            t = min(u1)

            length1 = e - t

            for i in range(len(d)):             #为了拟合边缘,将窗口中的绿色点按纵坐标分为7组,含坐标最多的组才用来拟合
                if 0 < v[i] - q < length / 7:
                    yi.append(d[i])
                elif length / 7 < v[i] - q < length *2/7 :
                    er.append(d[i])
                elif length *2/7 < v[i] - q < length *3/7:
                    san.append(d[i])
                elif length *3/7  < v[i] - q < length *4/7:

                    si.append(d[i])
                elif length *4/ 7 < v[i] - q < length *5/7 :
                    wu.append(d[i])
                elif length *5/ 7 < v[i] - q < length *6/7 :
                    liu.append(d[i])
                elif length *6/ 7 < v[i] - q < length  :
                    qi.append(d[i])





            changdu = [len(yi), len(er), len(san), len(si),len(wu),len(liu),len(qi)]




            if (max(changdu) == len(yi)):
                d=yi



            elif (max(changdu) == len(er)):
                d=er



            elif (max(changdu) == len(san)):
                d=san

            elif (max(changdu) == len(si)):
                d=si
            elif (max(changdu) == len(wu)):

                d=wu
            elif (max(changdu) == len(liu)):

                d=liu
            elif (max(changdu) == len(qi)):

                d=qi



            for i in range(len(d)):
                if 0 < u1[i] - t < length1 / 8:
                    yi1.append(d[i])
                elif length1 / 8 < u1[i] - t< length1*2 / 8:
                    er1.append(d[i])
                elif length1 *2/ 8 < u1[i] - t < length1 *3/ 8:

                    san1.append(d[i])
                elif length1 *3/ 8 < u1[i] - t < length1  *4/ 8:

                    si1.append(d[i])
                elif length1 *4 / 8 < u1[i] - t < length1  *5/ 8:

                    wu1.append(d[i])

                elif length1*5/ 8<u1[i]-t<length1*6/ 8:
                    liu1.append(d[i])
                elif length1*6/ 8<u1[i]-t<length1*7/8:
                    qi1.append(d[i])
                elif length1*7/8<u1[i]-t<length1:
                    ba1.append(d[i])


            changdu1 = [len(yi1), len(er1), len(san1), len(si1), len(wu1), len(liu1), len(qi1), len(ba1)]
            if (max(changdu1) == len(yi1)):
                d1 = yi1



            elif (max(changdu1) == len(er1)):
                d1 = er1



            elif (max(changdu1) == len(san1)):
                d1 = san1

            elif (max(changdu1) == len(si1)):
                d1 = si1
            elif (max(changdu1) == len(wu1)):
                d1 = wu1
            elif (max(changdu1) == len(liu1)):
                d1 = liu1
            elif (max(changdu1) == len(qi1)):
                d1 = qi1
            elif (max(changdu1) == len(ba1)):
                d1 = ba1




            changdu2=sorted(changdu1)
            #print(changdu2)

            if changdu2[1]==len(yi1):
                d2 =yi1

            elif changdu2[1]==len(er1):
                d2=er1
            elif changdu2[1]==len(san1):
                d2=san1
            elif changdu2[1]==len(si1):
                d2=si1
            elif changdu2[1]==len(wu1):
                d2=wu1
            elif changdu2[1]==len(liu1):
                d2=liu1
            elif changdu2[1]==len(qi1):
                d2=qi1
            elif changdu2[1]==len(ba1):
                d2=ba1




            '''else :

                d=liu'''




            #print(d)

            p=[]
            for i in range(len(d)):
                p.append((d[i][1],d[i][0]))



            sorted_point = sorted(p)
            u = [point[0] for point in sorted_point]
            v = [point[1] for point in sorted_point]

            p1 = []
            for i in range(len(d1)):
                p1.append((d1[i][1], d1[i][0]))
            sorted_point1 = sorted(p1)
            u3 = [point[0] for point in p1]
            v3 = [point[1] for point in p1]

            p2 = []
            for i in range(len(d2)):
                p2.append((d2[i][1], d2[i][0]))
            # sorted_point1 = sorted(p2)
            u4 = [point[0] for point in p2]
            v4 = [point[1] for point in p2]









            N = len(u)                                      #对获得的绿色像素点进行处理
            for i in range(int(len(u) / 2)):
                u[i], u[N - i - 1] = u[N - i - 1], u[i]

            M = len(v)
            for i in range(int(len(v) / 2)):
                v[i], v[M - i - 1] = v[M - i - 1], v[i]


            #for i in  range(len(u)):
                #v1=a*u+b

            a,b=Least_squares(u,v)#最小二乘法拟合直线,a是斜率.b是偏移
            print(a,b)
            print(a)
            zhixian.append(b)
            if a[0]>0.0500000 :      #将比较离谱的拟合直线取一个较常见的值

                a[0]=0.00200000
            elif a[0]<-0.0500000:
                a[0]=-0.00200000

            v1=a*u+b

            dian = []

            zhixian.append('%fx+%f'%(a,b))
            pianyi.append(b)
            #for i in range(slice_sets1):

            #plt.figure(figsize=(10, 5), facecolor='w')
            plt.figure(figsize=(10, 5))

            plt.plot(u, v, 'ro', lw=2, markersize=6)

            plt.plot(u, v1, 'b-', lw=2, markersize=6)
            #plot2=plt.plot(u)
            plt.grid(b=True, ls=':')

            plt.xlabel(u'X', fontsize=16)
            plt.ylabel(u'Y', fontsize=16)

            plt.imshow(slice1)
            v11 = []                             #在原图上画拟合直线

            for i in range(len(u)):
                v111 = int(a * u[i] + b)
                v11.append(v111)

            for l in range(len(u)):
                u[l] += x
            for k in range(len(v11)):
                v11[k] += y


            for l in range(len(u3)):
                u3[l] += x

            for k in range(len(v3)):
                v3[k] += y
            for l in range(len(u4)):
                u4[l] += x

            for k in range(len(v4)):
                v4[k] += y

            diyigedian = [point for point in p if point in p1]
            diergedian = [point for point in p if point in p2]






            if max(v3)-min(v3)>winH/3:            #竖边如果小于长度小于切片的一半就认为不是边缘而是干扰点
                cv2.line(img1, (u3[len(u3) - 1], v3[0]), (u3[len(u3) - 1], v3[len(v3) - 1]), (0, 0, 255), thickness=3, lineType=8)
            elif max(v3)-min(v3)<winH/3:
                v3.clear()


            '''if v4 and max(v4)-min(v4)<winH/2:
                v4.clear()'''
            if v4  and diyigedian:
                cv2.line(img1, (u4[len(u4) - 1], v4[0]), (u4[len(u4) - 1], v4[len(v4) - 1]), (0, 0, 255), thickness=3,
                         lineType=8)
                '''if u[0] != x+winW and c not in zuihouyige :
                    cv2.line(img1, (diyigedian[0]), (x, v11[len(v11) - 1]), (255, 0, 255), thickness=3, lineType=8)'''



            if  u[len(u)-1] !=x and c not in diyige and c not in zuihouyige and u[0]!=x+winW :         #根据切片位置进行优化
                cv2.line(img1, (x+winW, v11[0]), (x, v11[len(v11) - 1]), (255, 0, 255), thickness=3, lineType=8)
                vector1=np.array([x+winW, v11[0]])
                vector2=np.array([x, v11[len(v11) - 1]])
                op1 = np.sqrt(np.sum(np.square(vector1 - vector2)))
                changdu.append(op1)
            elif u[0] != x+winW and c not in zuihouyige :
                cv2.line(img1, (x+winW, v11[0]), (u[len(u) - 1], v11[len(v11) - 1]), (255, 255, 0), thickness=3, lineType=8)
                vector1 = np.array([x + winW, v11[0]])
                vector2 = np.array([u[len(u) - 1], v11[len(v11) - 1]])
                op1 = np.sqrt(np.sum(np.square(vector1 - vector2)))
                changdu.append(op1)
            elif u[len(u)-1] != x and c not in diyige:
                cv2.line(img1, (u[0], v11[0]), (x, v11[len(v11) - 1]), (255, 255, 255), thickness=3, lineType=8)
                vector1 = np.array([u[0], v11[0]])
                vector2 = np.array([x, v11[len(v11) - 1]])
                op1 = np.sqrt(np.sum(np.square(vector1 - vector2)))
                changdu.append(op1)
            else:
                cv2.line(img1, (u[0], v11[0]), (u[len(u) - 1], v11[len(v11) - 1]), (0, 0, 255), thickness=3, lineType=8)
                vector1 = np.array([u[0], v11[0]])
                vector2 = np.array([u[len(u) - 1], v11[len(v11) - 1]])
                op1 = np.sqrt(np.sum(np.square(vector1 - vector2)))
                changdu.append(op1)

            #if u[len(v11) - 1]==x + winW&i%6!=0:
            print("第%i块木板的长度:"%c,op1)
            cv2.imshow('nihe', img1)

            #slice1=plt.plot(u, v1, 'b-', lw=2, markersize=6)

            '''sp = slice.shape
            w = sp[0]  # height(rows) of image
            h = sp[1]
            color_size = sp[2]

            green_pix_ind = [0, 255, 0]
            red_pix_ind = [0, 0, 255]
            array=np.array(slice)
            for x in range(w):
                for y in range(h):


                    #bgr = slice1[x, y]

                    # m = int(a * x + b)
                    # if(y == int(a * x + b)) :

                    array[x, y] = red_pix_ind'''

            print('cnt',cnt)
            plt.show()

            #plt.close()

            print('c=',c)
            print('zhixian:',zhixian)
            print('pianyi:',pianyi)
            '''i += 1
            plt.savefig('/home/lk/Desktop/项目/效果图/test{}.png'.format(i))
            plt.clf()'''
            c=c+1
            #cv2.imshow('sdfds',img1)

    for i in range(len(pianyi) - 4):
        gaodu1 = H/22.5 - pianyi[i] + pianyi[i + 4]
        houdu.append(gaodu1)
    print('木板的厚度:', houdu)
    '''u=np.linspace(min(u),max(u),num=50)
            u = np.array(u)
            v = np.array(v)
            fl = np.polyfit(u, v, 2)  # 用3次多项式拟合
            #a0,a1 = linear_regression(u,v)
            pl = np.poly1d(fl)
            lv = pl(u)  # 拟合u值,不知道为什么画图u和v是反的
            print(pl)
            plot1 = plt.plot(u,v, 'r*')
            #plot2 = plt.plot(u , lv , 'b')
            plt.show()'''
        #jinghua(slice)

        #print(zhixian)



    cnt = cnt + 1

    cv2.imwrite('/home/lk/Desktop/项目/效果图/zhengfutunihe.jpg',img1)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
