import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from skimage.io import imread, imshow
from PIL import Image
from skimage.color import rgb2gray
from os.path import normpath as fn
# from matplotlib.image import imread
from mpl_toolkits.mplot3d import Axes3D
from skimage.transform import resize
import math
from scipy.optimize import leastsq
import glob as glob
import os
import time as time
from argparse import ArgumentParser
import cv2
import sklearn.metrics as skm

# https://blog.csdn.net/shandianfengfan/article/details/113706496
# -*- coding: utf-8  -*-
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#https://blog.csdn.net/shandianfengfan/article/details/119988689
# https://www.codeleading.com/article/8870348757/
def Draw_board(width, height, board_size):
    width_pix = (width + 1) * board_size  # + board_size  # add extra  board_size  for reserve blank
    height_pix = (height + 1) * board_size  # + board_size

    # white = (255,255,255)
    # black =  (0,0,0)

    image = np.zeros((height_pix, width_pix, 3), dtype=np.uint8)
    image.fill(255)

    color = (255, 255, 255)

    y0 = 0
    fill_color = 0
    for j in range(0, height + 1):
        y = j * board_size
        for i in range(0, width + 1):
            # rint(i)
            x0 = i * board_size
            y0 = y
            rect_start = (x0, y0)

            x1 = x0 + board_size
            y1 = y0 + board_size
            rect_end = (x1, y1)
            # print(x0, y0, x1, y1, fill_color)
            cv.rectangle(image, rect_start, rect_end, color, 1, 0)
            # print(fill_color)
            image[y0:y1, x0:x1] = fill_color
            if width % 2:
                if i != width:
                    fill_color = (0 if (fill_color == 255) else 255)
            else:
                if i != width + 1:
                    fill_color = (0 if (fill_color == 255) else 255)

    # image[0:20,0:20] = 0
    # image[40:60,0:20] = 0

    return image


def creat_bspline(rows,cols,grid_spacing):
    row_block_num, col_block_num = int(rows / grid_spacing), int(cols / grid_spacing)
    # 网格边界需要将图像边界包含
    BPLINE_BOARD_SIZE = 3
    # 网格尺寸大小
    grid_rows, grid_cols = row_block_num + BPLINE_BOARD_SIZE, col_block_num + BPLINE_BOARD_SIZE
    # 初始化网格点权重
    # grid_points=5*np.random.rand(2,grid_rows,grid_cols)
    grid_point = 10*np.random.random((2,grid_rows, grid_cols))

    deltx = rows * 1.0 / row_block_num
    delty = cols * 1.0 / col_block_num

    return [grid_point,deltx,delty ]

def pre_cal_bspline(indx,indy,deltx,delty):
    #计算图像中每个点在网格中的相对位置并做预计算
    x_block = (indx * 1.0 / deltx).astype(np.float)
    y_block = (indy * 1.0 / delty).astype(np.float)
    x_block_floor = np.floor(x_block)
    y_block_floor= np.floor(y_block)
    u = x_block - x_block_floor
    v = y_block - y_block_floor
    #为B样条函数计算做预准备
    u2 = u ** 2;
    u3 = u ** 3
    v2 = v ** 2;
    v3 = v ** 3
    #初始化B样条函数矩阵序列
    pX, pY = np.zeros((len(indx), 4, 1)), np.zeros((len(indy), 4, 1))
    pX[:, 0, 0] = (1 - u3 + 3 * u2 - 3 * u) / 6.0
    pX[:, 1, 0] = (4 + 3 * u3 - 6 * u2) / 6.0
    pX[:, 2, 0] = (1 - 3 * u3 + 3 * u2 + 3 * u) / 6.0
    pX[:, 3, 0] = u3 / 6.0
    pY[:, 0, 0] = (1 - v3 + 3 * v2 - 3 * v) / 6.0
    pY[:, 1, 0] = (4 + 3 * v3 - 6 * v2) / 6.0
    pY[:, 2, 0] = (1 - 3 * v3 + 3 * v2 + 3 * v) / 6.0
    pY[:, 3, 0] = v3 / 6.0
    return [pX,pY,x_block_floor,y_block_floor]

def bspline_trans(img,indx,indy,grid_point,pX,pY,x_block_floor,y_block_floor):
    #复制输入图像
    dstimg = img.copy()
    #计算每个点在对应网格中的相对位置
    # 计算tx,ty的系数,参考链接：https://www.zhihu.com/question/411657859
    coeff_xy = (pX @ np.transpose(pY, [0, 2, 1]))  # .reshape(len(indx),-1)
    #初始化每个坐标的相对位移
    Tx, Ty = np.zeros(len(indx)), np.zeros(len(indy))
    #计算每个点的相对位移
    for m in range(0, 4):
        control_point_x = (x_block_floor + m).astype(np.int)
        for n in range(0, 4):
            control_point_y = (y_block_floor + n).astype(np.int)
            grid_pointx_loc = grid_point[0,control_point_x, control_point_y]
            grid_pointy_loc = grid_point[1,control_point_x, control_point_y]
            Tx += coeff_xy[:, m, n] * grid_pointx_loc
            Ty += coeff_xy[:, m, n] * grid_pointy_loc

    #裁切去除掉超过边界的点
    fin_x = np.clip((np.floor(indx + Tx)).astype(np.int), 0, max(indx))
    fin_y = np.clip((np.floor(indy + Ty)).astype(np.int), 0, max(indy))
    #对新坐标进行赋值，这里采用的是最近邻法，当然可以采用双线性插值等算法
    dstimg[indx, indy] = img[fin_x, fin_y]

    return dstimg,Tx,Ty



if __name__ == "__main__":

    img = Draw_board(5, 5, 100)[:,:,2]
    rs, cs = img.shape
    x_ord, y_ord = np.where(img != np.inf)
    idx,idy = x_ord, y_ord

    gd_spacing=8
    gdpt, dx, dy = creat_bspline(rs, cs, gd_spacing)
    [px,py,xblock_floor,yblock_floor]=pre_cal_bspline(idx,idy,dx,dy)




    coeffxy = (px @ np.transpose(py, [0, 2, 1]))  # .reshape(len(indx),-1)
    coxy=coeffxy.reshape(rs,cs,4,4)


    #根据初始化得到的gridpoint计算对应的每个点的偏移量
    Tx1, Ty1 = np.zeros(len(idx)), np.zeros(len(idy))
    for m in range(0, 4):
        control_point_x = (xblock_floor + m).astype(np.int)
        for n in range(0, 4):
            control_point_y = (yblock_floor + n).astype(np.int)
            grid_pointx_loc = gdpt[0,control_point_x, control_point_y]
            grid_pointy_loc = gdpt[1,control_point_x, control_point_y]
            Tx1 += coeffxy[:, m, n] * grid_pointx_loc
            Ty1 += coeffxy[:, m, n] * grid_pointy_loc

    #Local update some grid's weights，Still need to calculate 4*4=16 points
    gdpt[:,27,17]+=[0.342,0.288]
    Tx2, Ty2 = np.zeros(len(idx)), np.zeros(len(idy))
    for m in range(0, 4):
        control_point_x = (xblock_floor + m).astype(np.int)
        for n in range(0, 4):
            control_point_y = (yblock_floor + n).astype(np.int)
            grid_pointx_loc = gdpt[0,control_point_x, control_point_y]
            grid_pointy_loc = gdpt[1,control_point_x, control_point_y]
            Tx2 += coeffxy[:, m, n] * grid_pointx_loc
            Ty2 += coeffxy[:, m, n] * grid_pointy_loc

    #Local update some grid's weights，Only need to calculate 1 points
    location=[27,17]
    bias=[0.342,0.288]
    #只计算部分区域的点值
    control_point_xx,control_point_yy=location
    x1=np.floor((control_point_xx-3)*dx)
    y1=np.floor((control_point_yy-3)*dy)
    x2=np.floor((control_point_xx+1)*dx)
    y2=np.floor((control_point_yy+1)*dy)
    result1=(np.clip([x1,x2],0,rs)).astype(np.int)
    result2=(np.clip([y1,y2],0,cs)).astype(np.int)
    grid_loc = []
    grid_loc.append(np.arange(result1[0]+1, result1[1], 1))#此处起始点+1是为了保持与实际计算误差区域一致
    grid_loc.append(np.arange(result2[0]+1, result2[1], 1))
    meshgrid = np.meshgrid(grid_loc[0], grid_loc[1])
    indxx=(np.squeeze((meshgrid[0]).T).reshape(-1)).astype(np.int)
    indyy=(np.squeeze((meshgrid[1]).T).reshape(-1)).astype(np.int)#得到这些点的索引列表

    x_block = (indxx * 1.0 / dx).astype(np.float)
    y_block = (indyy * 1.0 / dy).astype(np.float)
    x_blockfloor = (np.floor(x_block)).astype(np.int)
    y_blockfloor=  (np.floor(y_block)).astype(np.int)
    m=(location[0]-x_blockfloor).astype(np.int)
    n=(location[1]-y_blockfloor).astype(np.int)


    Tx3=(Tx1.copy()).reshape(rs,cs)
    Ty3=(Ty1.copy()).reshape(rs,cs)

    """感觉问题出在这里，把上面提到的直接赋值，改成了copy"""
    Tx3[indxx,indyy]+=coxy[indxx,indyy, m, n] *0.342#1是偏移量
    Ty3[indxx,indyy]+=coxy[indxx,indyy, m, n] * 0.288

    """tx1 is the original offset, tx2 is the new offset computed based on the B-spline localization property after updating one point randomly, 
    and tx3 is the new offset computed after optimizing b by computing only one control point, tx2=tx3"""
    errx1=np.sum(Tx1-Tx2)
    erry1=np.sum(Ty1-Ty2)
    errx2=np.sum(Tx1-Tx3.reshape(-1))
    erry2=np.sum(Ty1-Ty3.reshape(-1))
    errx3=np.sum(abs(Tx2-Tx3.reshape(-1)))
    erry3=np.sum(abs(Ty2-Ty3.reshape(-1)))


    abroke=1
