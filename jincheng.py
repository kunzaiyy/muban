import cv2
import numpy as np

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



def access_pixels(slice):
    print(slice.shape)
    a=[]
    h=slice.shape[0]
    w=slice.shape[1]
    print('weight : % s,height : % s'%(w,h))
    for row in range(h):
        for col in range(w):
            pv=slice[row,col]
            while(pv==255):
                a.append(pv)
    return pv
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

#def juhe(silce_sets):





if __name__ == '__main__':
    image = cv2.imread('/home/lk/Desktop/项目/裁剪后的/final.png')

    # 自定义滑动窗口的大小
    w = image.shape[1]
    h = image.shape[0]
    # 本代码将图片分为3×3，共九个子区域，winW, winH和stepSize可自行更改
    (winW, winH) = (int(w),int(h/10))
    stepSize = (int (w),int(h/10))
    cnt = 0
    for (x, y, window) in sliding_window(image, stepSize=stepSize, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        # since we do not have a classifier, we'll just draw the window
        clone = image.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        cv2.waitKey(1000)

        slice = image[y:y+winH,x:x+winW]

        cv2.namedWindow('sliding_slice',0)
        cv2.imshow('sliding_slice', slice)
        slice = handle_img(slice)

        '''b=np.array()
        b=np.concatenate((b,slice),axis = 2)
        cv2.imshow('sdf',b)

        cv2.waitKey(1000)'''





        cnt = cnt + 1
    '''windowSize=(winW,winH)
    get_slice(image,stepSize,windowSize)
    slice_sets = get_slice(image, stepSize, windowSize)
    for slice in slice_sets:
        access_pixels(slice)



        a=cv2.boxPoints()
        x,y,w,h=cv2.boudingRect'''
    '''slice_sets = get_slice(image,stepSize,windowSize)
    for slice in slice_sets:

        x, y = handle_img(img)'''