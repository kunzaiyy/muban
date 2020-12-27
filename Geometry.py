import os
import math
import numpy as np
import random
import copy
import time
import open3d as o3d
import cv2
import matplotlib.pyplot as plt

# resolution = [1080, 1920]
# intrinsic = [910.496, 910.197, 961.223, 556.013] # fx,fy,cx,cy
# scale = 0.001 / 1.057184072

resolution = [3072, 4096]
intrinsic = [1938.7,1938.7,2050.8, 1557.7] # fx,fy,cx,cy
scale = 0.001 / 1.057184072


# 直线拟合，剔除outline
# #todo 封装到新的文件中，如果拟合直接根据上一条线的斜率，那可以既快又准
def LineFitting(points, gap = 5):
    '''
        完成直线拟合
        先将事先认为是边缘的点拟合出直线,如果点到线的距离小于木板间距/3,就认为是内点,然后应用内点再次拟合为直线,再次循环,反复迭代3次
        :param points: 获取的所有属于木板边缘的点
        :param gap:木板间距
        :return:拟合直线k,b,内点数目
    '''
    line = cv2.fitLine(points, cv2.DIST_L2, 0, 1e-2, 1e-2)

    k = line[1] / line[0]
    b = line[3] - k * line[2]
    energy = 0
    inlier_ids = []
    inlier_num = 0
    max_iter = 3
    for it in range(max_iter):
        for i,point in enumerate(points): # 找内点
            dis = np.abs(k*point[0] - point[1] + b) / np.sqrt(k*k+1) #点到线的距离
            if dis < gap/3:
               inlier_ids.append(i)
               energy +=dis
        # print(len(ids), energy/len(ids))
        inlier_points = []
        for l,id in enumerate(inlier_ids):
            inlier_points.append(points[id])
        inlier_points = np.array(inlier_points)
        line = cv2.fitLine(inlier_points, cv2.DIST_L2, 0, 1e-2, 1e-2)
        k = line[1] / line[0]
        b = line[3] - k * line[2]

        inlier_num = len(inlier_ids)
        energy = 0
        inlier_ids.clear()
        #TODO: 中断迭代



    return k,b,inlier_num

# project to original color image to get colors of warped image
def WarpImg(rot, d, pos, size, color, depth):
    colorf = np.array(color, dtype='float')
    depthf = np.array(depth, dtype='float')
    img_color = np.zeros([size[0], size[1], 3], dtype='uint8')
    img_depth = np.zeros([size[0], size[1]], dtype='uint8')

    # 2d back-project to 3d
    Mu = np.zeros(size)
    for index in range(size[1]):
        Mu[:, index] = index
    Mv = np.zeros(size)
    for index in range(size[0]):
        Mv[index, :] = index
    z = d
    x = (Mu + pos[1] - intrinsic[2]) / intrinsic[0] * z
    y = (Mv + pos[0] - intrinsic[3]) / intrinsic[1] * z

    x = x.flatten()
    y = y.flatten()
    points = np.vstack((x, y))
    points = np.vstack((points, np.zeros(len(x))))
    points = np.transpose(points)

    # trans to original: R'*(p-t)+t
    points_origin = np.dot(np.transpose(rot), np.transpose(points))
    points_origin = np.transpose(points_origin)
    points_origin[:, 2] += z
    # project to original image
    pixels_origin = To2D(points_origin)

    #make sure all in image
    pixels_origin[:, 0][pixels_origin[:, 0] < 0] = 0
    pixels_origin[:, 0][pixels_origin[:, 0] > color.shape[1]-1] = color.shape[1] - 1
    pixels_origin[:, 1][pixels_origin[:, 1] < 0] = 0
    pixels_origin[:, 1][pixels_origin[:, 1] > color.shape[0] - 1] = color.shape[0] - 1

    # get up down left right pixel position
    u_left = np.floor(pixels_origin[:,0])
    u_right = np.ceil(pixels_origin[:,0])
    v_up = np.floor(pixels_origin[:,1])
    v_down = np.ceil(pixels_origin[:,1])

    u_left_w = u_right - pixels_origin[:,0]
    u_right_w = pixels_origin[:,0] - u_left
    v_up_w = v_down - pixels_origin[:,1]
    v_down_w = pixels_origin[:,1] - v_up

    u_left = u_left.astype(np.int16)
    u_right = u_right.astype(np.int16)
    v_up = v_up.astype(np.int16)
    v_down = v_down.astype(np.int16)

    timev1 = time.time()
    # get values
    for u in range(img_color.shape[1]):
        for v in range(img_color.shape[0]):
            id = v * img_color.shape[1] + u

            color_vmin = u_left_w[id] * colorf[v_up[id], u_left[id], :] \
                         + u_right_w[id] * colorf[v_up[id], u_right[id], :]
            color_vmax = u_left_w[id] * colorf[v_down[id], u_left[id], :] \
                         + u_right_w[id] * colorf[v_down[id], u_right[id], :]
            pt_color = v_up_w[id] * color_vmin \
                       + v_down_w[id] * color_vmax

            img_color[v, u, 0] = int(pt_color[0])
            img_color[v, u, 1] = int(pt_color[1])
            img_color[v, u, 2] = int(pt_color[2])

            depth_vmin = u_left_w[id] * depthf[v_up[id], u_left[id]] \
                         + u_right_w[id] * depthf[v_up[id], u_right[id]]
            depth_vmax = u_left_w[id] * depthf[v_down[id], u_left[id]] \
                         + u_right_w[id] * depthf[v_down[id], u_right[id]]
            pt_depth = v_up_w[id] * depth_vmin \
                       + v_down_w[id] * depth_vmax
            img_depth[v, u] = 255 if pt_depth > 128 else 0

    timev2 = time.time()
    print('time trans:', timev2-timev1)

    return img_color, img_depth


def GetWarpedImg(plane_model, plane_points, color):

    # project 3D points to image
    depth = np.zeros(resolution, dtype='uint8')
    color_debug = color.copy()

    uv = To2D(plane_points)
    time2 = time.time()
    uv = np.round(uv)
    uv = uv.astype(np.int16)
    uv[:, 0][uv[:, 0] < 0] = 0
    uv[:, 0][uv[:, 0] > color.shape[1]-1] = color.shape[1] - 1
    uv[:, 1][uv[:, 1] < 0] = 0
    uv[:, 1][uv[:, 1] > color.shape[0] - 1] = color.shape[0] - 1
    for value in uv:#TODO 耗时14s
        color_debug[value[1],value[0],:] = [128,128,128]
        depth[value[1], value[0]] = 255

    time3 = time.time()
    print(time3-time2)
    plt.imshow(color_debug)
    plt.title('color with mask')  # 图像题目
    plt.show()

    #------------------------ 将平面旋转为与xy面平行 p'=R*(p-t)+t---------------------------
    z_offset = -plane_model[3] / plane_model[2]
    plane_points = plane_points - [0, 0, z_offset]
    ##--rot--Rodrigues Rotation Formula
    normal = plane_model[0:3]
    axis = np.cross(normal, [0,0,1])
    axis = np.array(axis)
    axis = axis / np.sqrt(axis[0]**2+axis[1]**2+axis[2]**2)
    cos = normal[2]
    sin = np.sqrt(normal[0] **2 + normal[1]**2)
    lx = [[0, -axis[2], axis[1]],
          [axis[2], 0, -axis[0]],
          [-axis[1], axis[0], 0]]
    lx = np.array(lx)
    rot = np.identity(3) + (1 - cos) * lx**2 + sin * lx
    rot[0,0] = cos
    rot[1,1] = cos
    rot[2,2] = 1.0

    #trans
    trans_points = np.transpose(np.dot(rot, np.transpose(plane_points)))
    trans_points = trans_points + [0, 0, z_offset]

    #------------------------ 将平面底边旋转为与x轴平行 p''=R*p'---------------------------
    #project to image to find bottom line
    img  = np.zeros(resolution,dtype='uint8')
    uv_trans = To2D(trans_points)
    uv_trans = np.round(uv_trans)
    uv_trans = uv_trans.astype(np.int16)
    uv_trans[:, 0][uv_trans[:, 0] < 0] = 0
    uv_trans[:, 0][uv_trans[:, 0] > color.shape[1]-1] = color.shape[1] - 1
    uv_trans[:, 1][uv_trans[:, 1] < 0] = 0
    uv_trans[:, 1][uv_trans[:, 1] > color.shape[0] - 1] = color.shape[0] - 1
    for value in uv_trans:
        img[value[1], value[0]] = 255

    pt = []
    for u in range(int(resolution[1]/3), int(resolution[1]/3*2),2):
        for v in range(int(resolution[0]-1),int(resolution[0]/2),-2):
            if img[v,u] == 255:
                pt.append([u,v])
                break
    pt = np.array(pt)
    k,b,ine = LineFitting(pt, color.shape[0] / 100)

    cv2.line(img, (0,int(b)), ((img.shape[1]-1),int((img.shape[1]-1)*k+b)) , 128, 2, cv2.LINE_AA)

    plt.imshow(img, cmap='gray')
    plt.title("bottomLineFit")  # 图像题目
    plt.show()



    #rot
    axis = [0,0,-1]
    cos = 1 / np.sqrt(1+k**2)
    sin = k / np.sqrt(1+k**2)
    lx = [[0, -axis[2], axis[1]],
          [axis[2], 0, -axis[0]],
          [-axis[1], axis[0], 0]]
    lx = np.array(lx)
    rot2 = np.identity(3) + (1 - cos) * lx**2 + sin * lx
    rot2[0,0] = cos
    rot2[1,1] = cos
    rot2[2,2] = 1.0

    #transfer
    trans_points = trans_points - [0, 0, z_offset]
    trans_points = np.transpose(np.dot(rot2, np.transpose(trans_points)))
    trans_points = trans_points + [0, 0, z_offset]



    #-----------------------------------------------------
    D = np.mean(trans_points[:,2])
    arr_std = np.max(trans_points[:,2]-D)
    print("平面深度的最大偏差:", arr_std)

    #visualize
    # trans_cloud = o3d.geometry.PointCloud()
    # trans_cloud.points = o3d.utility.Vector3dVector(trans_points)
    # trans_cloud.paint_uniform_color([1.0, 0, 0.0])
    # o3d.visualization.draw_geometries([trans_cloud],
    #                                   zoom=0.2,
    #                                   front=[0,0,-1],
    #                                   lookat=[0,0,0],
    #                                   up=[0,-1,0])

    #-----------------------------Get warp img size----------------
    x_max = np.max(trans_points[:,0])
    x_min = np.min(trans_points[:,0])
    y_max = np.max(trans_points[:,1])
    y_min = np.min(trans_points[:,1])
    u_max = int(np.round(x_max / D * intrinsic[0] + intrinsic[2]))
    u_min = int(np.round(x_min / D * intrinsic[0] + intrinsic[2]))
    v_max = int(np.round(y_max / D * intrinsic[1] + intrinsic[3]))
    v_min = int(np.round(y_min / D * intrinsic[1] + intrinsic[3]))

    u_min = 0 if u_min < 0 else u_min
    u_max = color.shape[1]-1 if u_max > color.shape[1]-1 else u_max
    v_min = 0 if v_min < 0 else v_min
    v_max = color.shape[0] - 1 if v_max > color.shape[0] - 1 else v_max

    print(u_min,' ', u_max)
    # 剔除地面的影响
    thre = color.shape[0] / 50 #TODO 调参 3000/50=60
    for ul in range(int(u_min), int(color.shape[1]/2), 1):
        pt_inliers = np.where(img[:, ul] == 255)[0]
        if len(pt_inliers) > thre:
            u_min = ul
            break
    for ur in range(int(u_max), int(color.shape[1]/2), -1):
        pt_inliers = np.where(img[:, ur] == 255)[0]
        if len(pt_inliers) > thre:
            u_max = ur
            break
    print(u_min, ' ', u_max)
    offset_bound = int(color.shape[0] / 100) #参数4000/100=40

    if u_max + offset_bound > resolution[1] - 1:
        u_max = resolution[1] - 1
    else:
        u_max = u_max + offset_bound

    if u_min - offset_bound < 0:
        u_min = 0
    else:
        u_min = u_min - offset_bound

    if v_max + offset_bound > resolution[0] - 1:
        v_max = resolution[0] - 1
    else:
        v_max = v_max + offset_bound

    if v_min - offset_bound < 0:
        v_min = 0
    else:
        v_min = v_min - offset_bound


    crop_img_size = [v_max-v_min+1, u_max-u_min+1]
    crop_img_pos = [v_min, u_min]

    img, imgd = WarpImg(np.dot(rot2, rot), D, crop_img_pos, crop_img_size, color, depth)
    # img, imgd = WarpImg(rot, D, crop_img_pos, crop_img_size, color, depth)
    return img, imgd, D







def PlaneFitting(points, seg_thre = 0.1, ransac_n = 10, iter_num = 10, clu_eps = 0.02, clu_minnum = 100):

    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)

    plane_model, inliers = pointcloud.segment_plane(seg_thre,
                                             ransac_n,
                                             iter_num)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a}x + {b}y + {c}z + {d} = 0")
    inlier_cloud = pointcloud.select_by_index(inliers)

    # o3d.visualization.draw_geometries([inlier_cloud],
    #                                   zoom=0.2,
    #                                   front=[0,0,-1],
    #                                   lookat=[0,0,0],
    #                                   up=[0,-1,0])

    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as plt.cm:
    #     labels = np.array(inlier_cloud.cluster_dbscan(clu_eps, clu_minnum))
    #
    # med = np.median(labels)
    # pt_id_inliers = np.where(labels == med)[0]
    # inlier_cloud = inlier_cloud.select_by_index(pt_id_inliers)

    #------------------------ Projection points to the plane------------------------

    pts = inlier_cloud.points
    pts = np.asarray(pts)
    plane_points = pts
    plane_points[:,0] = ((b * b + c * c) * pts[:,0] - a * (b * pts[:,1] + c * pts[:,2] + d)) / (a * a + b * b + c * c)
    plane_points[:,1] = ((a * a + c * c) * pts[:,1] - b * (a * pts[:,0] + c * pts[:,2] + d)) / (a * a + b * b + c * c)
    plane_points[:,2] = ((b * b + a * a) * pts[:,2] - c * (a * plane_points[:,0] + b * pts[:,1] + d)) / (a * a + b * b + c * c)

    inlier_cloud.points = o3d.utility.Vector3dVector(plane_points)

    # #visualize
    # inlier_cloud.paint_uniform_color([0, 0, 1.0])
    # o3d.visualization.draw_geometries([inlier_cloud],
    #                                   zoom=0.2,
    #                                   front=[0,0,-1],
    #                                   lookat=[0,0,0],
    #                                   up=[0,-1,0])


    return plane_model, plane_points


def To2D(points):
    # u = point[0] / point[2] * intrinsic[0] + intrinsic[2]
    # v = point[1] / point[2] * intrinsic[1] + intrinsic[3]

    uv = points[:, 0:2].copy()
    uv[:, 0] = uv[:, 0] / points[:, 2] * intrinsic[0] + intrinsic[2]
    uv[:, 1] = uv[:, 1] / points[:, 2] * intrinsic[1] + intrinsic[3]

    return uv




def BackProjection(img):
    points = []

    #accelerate
    Mu = np.zeros(img.shape)
    for index in range(img.shape[1]):
        Mu[:, index] = index
    Mv = np.zeros(img.shape)
    for index in range(img.shape[0]):
        Mv[index,:] = index

    z = img * scale
    x = np.multiply((Mu - intrinsic[2]) / intrinsic[0], z)
    y = np.multiply((Mv - intrinsic[3]) / intrinsic[1], z)

    z = z.flatten()
    x = x.flatten()
    y = y.flatten()

    points = np.vstack((x,y))
    points = np.vstack((points,z))
    points = np.transpose(points)

    # for u in range(img.shape[1]):
    #     for v in range(img.shape[0]):
    #         d = img[v,u]
    #         if not d:
    #             continue
    #         z = d * scale
    #         x = (u - intrinsic[2]) / intrinsic[0] * z
    #         y = (v - intrinsic[3]) / intrinsic[1] * z
    #         point = [x, y, z]
    #         points.append(point)
    return np.array(points)
