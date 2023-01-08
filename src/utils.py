from collections import deque
import numpy as np
from PIL import Image
import copy
import cv2 as cv
import math

def distance(i, j, x, y):
    return (i - x)**2 + (j - y)**2

def calc_rgb_l2(rgb1, rgb2):
    return (rgb1[0] - rgb2[0])**2 + (rgb1[1] - rgb2[1])**2 + (rgb1[2] - rgb2[2])**2

def morph_shape(val):
    if val == 0:
        return cv.MORPH_RECT
    elif val == 1:
        return cv.MORPH_CROSS
    elif val == 2:
        return cv.MORPH_ELLIPSE
    
def dilatation(image):
    src = cv.imread(cv.samples.findFile(image))
    if src is None:
        print('Could not open or find the image: ', image)
        exit(0)
    dilatation_size = 16
    dilation_shape = cv.MORPH_ELLIPSE
    # print("which?{}".format(cv.getTrackbarPos(title_trackbar_element_shape, title_dilation_window)))
    element = cv.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    dilatation_dst = cv.dilate(src, element)
    # print(dilatation_dst.shape)
    Image.fromarray((dilatation_dst).astype(np.uint8)).save("xxx.jpg")

def bfs_get_kernel(masked_path):
    pic = np.asarray(Image.open(masked_path))[...,:3]
    shape = pic.shape
    raw = shape[0]
    col = shape[1]
    output = np.zeros((raw, col, 3))
    black_queue = deque()
    for i in range(raw):
        for j in range(col):
            if pic[i][j][0] == 0 and pic[i][j][1] == 0 and pic[i][j][2] == 0:
                black_queue.append((i, j))
    edge_black_queue = deque()
    for xy in black_queue:
        x, y = xy
        if x + 1 < raw and not (pic[x+1][y][0] == 0 and pic[x+1][y][1] == 0 and pic[x+1][y][2] == 0):
            edge_black_queue.append((x, y))
        elif x - 1 >= 0 and not (pic[x-1][y][0] == 0 and pic[x-1][y][1] == 0 and pic[x-1][y][2] == 0):
            edge_black_queue.append((x, y))
        elif y + 1 < col and not (pic[x][y+1][0] == 0 and pic[x][y+1][1] == 0 and pic[x][y+1][2] == 0):
            edge_black_queue.append((x, y))
        elif y - 1 >= 0 and not (pic[x][y-1][0] == 0 and pic[x][y-1][1] == 0 and pic[x][y-1][2] == 0):
            edge_black_queue.append((x, y))
    
    for xy in edge_black_queue:
        x, y = xy
        output[x][y] = [255, 255, 255]
    Image.fromarray((output).astype(np.uint8)).save("tmp.jpg")
    
    dilatation("tmp.jpg")
    
    pic_b = np.zeros((raw, col, 3))
    tmp = np.asarray(Image.open("xxx.jpg"))[...,:3]
    for i in range(raw):
        for j in range(col):
            if (tmp[i][j] >= [200, 200, 200]).all():
                pic_b[i][j] = pic[i][j]
    up = -1
    down = raw
    left = -1
    right = col
    for r in range(raw):
        if up != -1:
            break
        for c in range(col):
            if pic_b[r][c][0] > 20 and pic_b[r][c][1] > 20 and pic_b[r][c][2] > 20:
                up = max(r-1, 0)
                break
    for r in range(raw):
        if down != raw:
            break
        for c in range(col):
            if pic_b[raw-1-r][c][0] > 20 and pic_b[raw-1-r][c][1] > 20 and pic_b[raw-1-r][c][2] > 20:
                down = min(raw-r, raw-1)
                break
    
    for c in range(col):
        if left != -1:
            break
        for r in range(raw):
            if pic_b[r][c][0] > 20 and pic_b[r][c][1] > 20 and pic_b[r][c][2] > 20:
                left = max(c-1, 0)
                break
    for c in range(col):
        if right != col:
            break
        for r in range(raw):
            if pic_b[r][col-1-c][0] > 20 and pic_b[r][col-1-c][1] > 20 and pic_b[r][col-1-c][2] > 20:
                right = min(col-c, col-1)
                break
    kernel = np.zeros((down - up + 1, right - left + 1, 3))
    print(kernel.shape)
    for r in range(up, down + 1):
        for c in range(left, right + 1):
            kernel[r - up][c - left] = pic_b[r][c]
    Image.fromarray((kernel).astype(np.uint8)).save("kernel.jpg")
    return kernel, up, left


def calcL2(A, kernel, offset_r, offset_c): # A是原图。计算这个位置下的L2误差
    shape_a = A.shape
    # row_a = shape_a[0]
    # col_a = shape_a[1]
    
    shape_kernel = kernel.shape
    row_kernel = shape_kernel[0]
    col_kernel = shape_kernel[1]
    L2 = 0
    for r in range(row_kernel):
        for c in range(col_kernel):
            r_a = r + offset_r
            c_a = c + offset_c
            if kernel[r][c][0] > 25 and kernel[r][c][1] > 25 and kernel[r][c][2] > 25: # kernel不是黑的
                rgb_kernel = kernel[r][c]
                rgb_a = A[r_a][c_a]
                L2 += calc_rgb_l2(rgb_kernel, rgb_a)
    return L2
        
def bestLocOffset(A, kernel):
    shape_a = A.shape
    row_a = shape_a[0]
    col_a = shape_a[1]
    
    shape_kernel = kernel.shape
    row_kernel = shape_kernel[0]
    col_kernel = shape_kernel[1]
    offset_r_max = row_a - row_kernel # 最大的能取到的offset
    offset_c_max = col_a - col_kernel
    
    min_l2 = 100000000
    min_o_r = 0
    min_o_c = 0
    for o_r in range(offset_r_max + 1):
        for o_c in range(offset_c_max + 1):
            L2 = calcL2(A, kernel, o_r, o_c)
            if L2 < min_l2:
                min_l2 = L2
                min_o_r = o_r
                min_o_c = o_c
    return min_o_r, min_o_c

def calcConvJittor(A, kernel):
    print(kernel.shape)
    print(A.shape)
    a_r = A.shape[0]
    a_c = A.shape[1]
    k_r = kernel.shape[0]
    k_c = kernel.shape[1]
    offset_r_max = a_r - k_r + 1
    offset_c_max = a_c - k_c + 1
    y = np.zeros([a_r - k_r + 1, a_c - k_c + 1, 1])
    # for i0 in range(offset_r_max):
    #     for i1 in range(offset_c_max):
    #         for i2 in range(k_r):
    #             for i3 in range(k_c):
    #                 for i4 in range(3):
    #                    # if i0 + i2 > a_r or i1 + i3 > a_c: continue
    #                     # print(kernel[i2, i3, i4])
    #                     y[i0, i1] += (kernel[i2, i3, i4] - A[i0 + i2, i1 + i3, i4])**2
    aa = A.reindex([offset_r_max, offset_c_max, k_r, k_c, 3], [
        'i0+i2',
        'i1+i3',
        'i4'
    ])
    kk = kernel.broadcast_var(aa)
    yy = (kk-aa)*(kk-aa)*(kk!=0)
    y = yy.sum([2, 3, 4])
    return y
    
    
def adjacent(x, y, a, b):
    return (x-1==a and y==b) or (x+1==a and y==b) or (x==a and y-1==b) or (x==a and y+1==b)

def calcMatrixA(res_list: list):
    N = res_list.__len__()
    print("size:{}".format(N))
    MaxtrixA = np.zeros((N, N, ))
    for i in range(N):
        for j in range(N):
            if i == j:
                MaxtrixA[i][j] = 4
            elif adjacent(res_list[i][0], res_list[i][1], res_list[j][0], res_list[j][1]):
                MaxtrixA[i][j] = -1
            else:
                MaxtrixA[i][j] = 0
    return MaxtrixA

def calcB(res_list, source_pic, target_pic, source_offset_r, source_offset_c, target_offset_r, target_offset_c):
    N = res_list.__len__()
    ret_r = []
    ret_g = []
    ret_b = []
    for i in range(N):
        x, y = res_list[i]
        row = source_pic.shape[0]
        col = source_pic.shape[1]
        rgb = target_pic[x-source_offset_r+target_offset_r][y-source_offset_c+target_offset_c].astype(np.int32)
        adj_list = []
        if x+1 < row:
            adj_list.append((x+1, y))
        if x-1 >= 0:
            adj_list.append((x-1, y))        
        if y+1 < col:
            adj_list.append((x, y+1))
        if y-1 >= 0:
            adj_list.append((x, y-1))
        tmp = 4 * rgb
        for ab in adj_list:
            a, b = ab
            tmp -= target_pic[a-source_offset_r+target_offset_r][b-source_offset_c+target_offset_c].astype(np.int32)
            if (a, b) not in res_list:
                tmp += source_pic[a][b].astype(np.int32) 
        
        ret_r.append(tmp[0])
        ret_g.append(tmp[1])
        ret_b.append(tmp[2])
    return ret_r, ret_g, ret_b # 返回三个颜色通道的b向量
        
        
    

    