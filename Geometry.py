import os
import math
import numpy as np
import random
import copy

import open3d as o3d
import cv2
import matplotlib.pyplot as plt

resolution = [1080, 1920]
intrinsic = [910.496, 910.197, 961.223, 556.013] # fx,fy,cx,cy
scale = 1000.0


# project to original color image to get colors of warped image
def WarpImg(rot, d, pos, size, color):
    colorf = np.array(color, dtype='float')
    img = np.zeros([size[0], size[1], 3], dtype='uint8')

    z = d
    for u in range(img.shape[1]):
        for v in range(img.shape[0]):
            x = (u + pos[1] - intrinsic[2]) / intrinsic[0] * z
            y = (v + pos[0] - intrinsic[3]) / intrinsic[1] * z
            point = [x, y, z]

            # trans to original: R'*(p-t)+t
            point[2] -= z
            p_ori = np.dot(np.transpose(rot), np.transpose(point))
            p_ori[2] += z

            pt_ori = To2D(p_ori)

            #TODO 防止超出图片大小
            umin = np.floor(pt_ori[0])
            umax = np.ceil(pt_ori[0])
            vmin = np.floor(pt_ori[1])
            vmax = np.ceil(pt_ori[1])

            color_vmin = (umax - pt_ori[0]) * colorf[int(vmin), int(umin), :] \
                        + (pt_ori[0] - umin) * colorf[int(vmin), int(umax), :]

            color_vmax = (umax - pt_ori[0]) * colorf[int(vmax), int(umin), :] \
                        + (pt_ori[0] - umin) * colorf[int(vmax), int(umax), :]

            pt_color =  (vmax - pt_ori[1]) * color_vmin \
                        + (pt_ori[1] - vmin) * color_vmax
            img[v,u,:] = np.array(pt_color,dtype='uint8')

    return img


def GetWarpedImg(plane_model, plane_points, color):

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
    for i, value in enumerate(trans_points):
        u, v = To2D(value)
        u = int(np.round(u))
        v = int(np.round(v))
        img[v, u] = 255

    pt = []
    for u in range(int(resolution[1]/3), int(resolution[1]/3*2),1):
        for v in range(int(resolution[0]-1),int(resolution[0]/2),-1):
            if img[v,u] == 255:
                pt.append([u,v])
                break
    pt = np.array(pt)

    line = cv2.fitLine(pt, cv2.DIST_L2, 0, 1e-2, 1e-2)
    k = line[1] / line[0]

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

    D = np.mean(trans_points[:,2])

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
    u_max = x_max / D * intrinsic[0] + intrinsic[2]
    u_min = x_min / D * intrinsic[0] + intrinsic[2]
    v_max = y_max / D * intrinsic[1] + intrinsic[3]
    v_min = y_min / D * intrinsic[1] + intrinsic[3]

    offset_bound = 20

    if u_max + offset_bound > resolution[1] - 1:
        u_max = resolution[1] - 1
    else:
        u_max = int(np.ceil(u_max)) + offset_bound

    if u_min - offset_bound < 0:
        u_min = 0
    else:
        u_min = int(np.floor(u_min)) - offset_bound

    if v_max + offset_bound > resolution[0] - 1:
        v_max = resolution[0] - 1
    else:
        v_max = int(np.ceil(v_max)) + offset_bound

    if v_min - offset_bound < 0:
        v_min = 0
    else:
        v_min = int(np.floor(v_min)) - offset_bound


    crop_img_size = [v_max-v_min+1, u_max-u_min+1]
    crop_img_pos = [v_min, u_min]

    img = WarpImg(np.dot(rot2, rot), D, crop_img_pos, crop_img_size, color)

    return img, D







def PlaneFitting(points, seg_thre = 0.1, ransac_n = 10, iter_num = 1000, clu_eps = 0.02, clu_minnum = 100):

    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)

    plane_model, inliers = pointcloud.segment_plane(seg_thre,
                                             ransac_n,
                                             iter_num)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a}x + {b}y + {c}z + {d} = 0")
    inlier_cloud = pointcloud.select_by_index(inliers)



    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as plt.cm:
        labels = np.array(inlier_cloud.cluster_dbscan(clu_eps, clu_minnum))

    med = np.median(labels)
    pt_id_inliers = np.where(labels == med)[0]
    inlier_cloud = inlier_cloud.select_by_index(pt_id_inliers)

    #------------------------ Projection points to the plane------------------------
    plane_points = []
    for i,pt in enumerate(inlier_cloud.points):
        x = ((b * b + c * c) * pt[0] - a * (b * pt[1] + c * pt[2] + d)) / (a * a + b * b + c * c)
        y = ((a * a + c * c) * pt[1] - b * (a * pt[0] + c * pt[2] + d)) / (a * a + b * b + c * c)
        z = ((b * b + a * a) * pt[2] - c * (a * x + b * pt[1] + d)) / (a * a + b * b + c * c)
        plane_points.append([x,y,z])
    plane_points = np.array(plane_points)
    inlier_cloud.points = o3d.utility.Vector3dVector(plane_points)

    #visualize
    # inlier_cloud.paint_uniform_color([0, 0, 1.0])
    # o3d.visualization.draw_geometries([inlier_cloud],
    #                                   zoom=0.2,
    #                                   front=[0,0,-1],
    #                                   lookat=[0,0,0],
    #                                   up=[0,-1,0])


    return plane_model, plane_points


def To2D(point):
    u = point[0] / point[2] * intrinsic[0] + intrinsic[2]
    v = point[1] / point[2] * intrinsic[1] + intrinsic[3]
    return u,v


def BackProjection(img):
    points = []

    for u in range(img.shape[1]):
        for v in range(img.shape[0]):
            d = img[v,u]
            if not d:
                continue
            z = d / scale
            x = (u - intrinsic[2]) / intrinsic[0] * z
            y = (v - intrinsic[3]) / intrinsic[1] * z
            point = [x, y, z]
            points.append(point)
    return np.array(points)
