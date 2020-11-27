import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

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

    #cv2.imshow("handle_img", image)
    #cv2.imwrite("C:/Users/ASUS/Desktop/2.png", src)
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
def access_pixels(slice):
    print(slice.shape)
    a=[]
    h=slice.shape[0]
    w=slice.shape[1]
    c=slice.shape[2]
    green_pix_ind=[0,255,0]
    print('weight : % s,height : % s'%(w,h))
    for row in range(h):
        for col in range(w):
            for channel in range(c):
                if( (Img[(i,j)][:]==green_pix_ind ).all() ):
                    pv=slice[row,col,c]
                    while(pv==255):
                        a.append(pv)
    return a

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

def yuchuli(v):





    return v



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




if __name__ == '__main__':
    #image = cv2.imread('/home/lk/Desktop/项目/裁剪后的/thresh.png')
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
    image=img
    # 自定义滑动窗口的大小
    w = image.shape[1]
    h = image.shape[0]
    # 本代码将图片分为3×3，共九个子区域，winW, winH和stepSize可自行更改
    (winW, winH) = (int(w/4),int(h/20))
    stepSize = (int (w/4),int(h/20))
    cnt = 0
    for (x, y, window) in sliding_window(image, stepSize=stepSize, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        # since we do not have a classifier, we'll just draw the window
        clone = image.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 0, 255), 2)
        cv2.imshow("Window", clone)


        slice = image[y:y+winH,x:x+winW]

        cv2.namedWindow('sliding_slice',0)
        cv2.imshow('sliding_slice', slice)
        slice_sets = []
        slice_sets.append(slice)

        for slice in slice_sets:

            a,b,d=green_points(slice)

            v = a
            u = b
            '''points = zip(u, v)
            print(points)
            sorted_points = sorted(points)
            u = [point[0] for point in sorted_points]
            v = [point[1] for point in sorted_points]'''
            z = max(v)
            q = min(v)
            length = z - q
            yi = []
            er = []
            san = []
            si = []
            wu=[]
            liu=[]
            
            for i in range(len(d)):
                if 0 < v[i] - q < length / 6:
                    yi.append(d[i])
                elif length / 6 < v[i] - q < length / 3:
                    er.append(d[i])
                elif length / 3 < v[i] - q < length /2:

                    san.append(d[i])
                elif length / 2 < v[i] - q < length * 2 / 3:


                    si.append(d[i])
                elif length *2/ 3 < v[i] - q < length * 5 / 6:

                    wu.append(d[i])

                else:
                    liu.append(d[i])
            #print(len(yi))
            #print(len(er))
            #print(len(san))
            changdu = [len(yi), len(er), len(san), len(si),len(wu),len(liu)]

            '''for i in range(len(d)):
                if 0 < u[i] - q < length / 4:
                    yi.append(d[i])
                elif length / 4 < u[i] - q < length / 2:
                    er.append(d[i])
                elif length / 2 < u[i] - q < length * 3 / 4:

                    san.append(d[i])
                else:
                    si.append(d[i])
            print(len(yi))
            print(len(er))
            print(len(san))
            changdu = [len(yi), len(er), len(san), len(si)]'''

            if (max(changdu) == len(yi)):
                d=yi
                    # while (len(yi)>len(er)|len(yi)>len(san)|len(yi)>len(si)):
                '''for k in range(len(d)):
                    for p in range(len(yi)):
                        if v[k] == yi[p]:
                            point.append(d[k])'''


            elif (max(changdu) == len(er)):
                d=er
                    # while (len(yi)>len(er)|len(yi)>len(san)|len(yi)>len(si)):
                '''for k in range(len(d)):
                    for p in range(len(er)):
                        if v[k] == er[p]:
                            point.append(d[k])'''


            elif (max(changdu) == len(san)):
                d=san
                    # while (len(yi)>len(er)|len(yi)>len(san)|len(yi)>len(si)):
                '''for k in range(len(d)):
                    for p in range(len(san)):
                        if v[k] == san[p]:
                            point.append(d[k])'''
            elif (max(changdu) == len(si)):
                d=si
            elif (max(changdu) == len(wu)):
                d=wu

            else :

                d=liu
                    # while (len(yi)>len(er)|len(yi)>len(san)|len(yi)>len(si)):
                '''for k in range(len(d)):
                    for p in range(len(si)):
                        if v[k] == si[p]:
                            point.append(d[k])'''



            #print(d)

            p=[]
            for i in range(len(d)):
                p.append((d[i][1],d[i][0]))
            sorted_point = sorted(p)
            u = [point[0] for point in sorted_point]
            v = [point[1] for point in sorted_point]




            N = len(u)
            for i in range(int(len(u) / 2)):
                u[i], u[N - i - 1] = u[N - i - 1], u[i]

            M = len(v)
            for i in range(int(len(v) / 2)):
                v[i], v[M - i - 1] = v[M - i - 1], v[i]



            a,b=Least_squares(u,v)
            print(a,b)
            v1=a*u+b
            plt.figure(figsize=(10, 5), facecolor='w')
            plt.plot(u, v, 'ro', lw=2, markersize=6)
            plt.plot(u, v1, 'b-', lw=2, markersize=6)
            plt.grid(b=True, ls=':')
            plt.xlabel(u'X', fontsize=16)
            plt.ylabel(u'Y', fontsize=16)
            plt.show()
            #v=yuchuli(v)
            #u=np.linspace(min(u),max(u),num=50)
            '''u = np.array(u)
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








        cnt = cnt + 1
        cv2.waitKey(1000)
        cv2.destroyAllWindows()




