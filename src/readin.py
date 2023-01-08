import numpy as np
from PIL import Image
import utils
import jittor as jt
from graph import buildGraph
import jparse

subdir = "../数据/input/"
dir_num = 1

pic_name_pref = "result_img"
pic_num = 1


def read_and_mask(filepath, maskpath, targetpath):
    src = np.asarray(Image.open(filepath))[...,:3]
    mask = np.asarray(Image.open(maskpath))[...,:3]
    shape = src.shape
    raw = shape[0]
    col = shape[1]
    output = np.zeros((raw, col, 3))
    for i in range(raw):
        for j in range(col):
            ori_rgb = src[i][j]
            mask_rgb = mask[i][j]
            if mask_rgb[0] < 20 and mask_rgb[1] < 20 and mask_rgb[2] < 20:
                output[i][j] = [0, 0, 0]
            else:
                output[i][j] = ori_rgb
    Image.fromarray((output).astype(np.uint8)).save(targetpath)


def calcConv(filepath, maskedpath):
    A = np.asarray(Image.open(filepath))[...,:3]
    kernel, ori_offset_r, ori_offset_c = utils.dila_get_kernel(maskedpath) #得到kernel
    js = jt.array(A)
    jk = jt.array(kernel)
    jy = utils.calcConvJittor(js, jk)# 计算卷积
    return jy, kernel, ori_offset_r, ori_offset_c





def main(down_input_path, down_input_mask_path, masked_path, result_path, output_path, type):

    read_and_mask(down_input_path, down_input_mask_path, masked_path)

    y, kernel, ori_offset_r, ori_offset_c = calcConv(result_path, masked_path) # 这里顺便返回了kernel在原图上的位置，用于做坐标变换。
    print(y.shape)

    y_row = y.shape[0]
    y_col = y.shape[1]
    min = 10000000000
    offset_r = -1
    offset_c = -1
    for r in range(y_row): # 拿到卷积计算后的矩阵，找到最小的那个位置，其位置代表了在候选图像上的偏移量offset，用于kernel坐标和候选坐标的转换
        for c in range(y_col):
            # print(y[r][c])
            if y[r][c] < min:
                min = y[r][c]
                offset_r = r
                offset_c = c

    # 建图并且计算graph-cut，即最小割
    partition = buildGraph(offset_r, offset_c, ori_offset_r, ori_offset_c, down_input_path , result_path, kernel, type=type)

    source, dest = partition

    origin = np.asarray(Image.open(down_input_path))[...,:3]
    mask = np.asarray(Image.open(masked_path))[...,:3]
    res = np.asarray(Image.open(result_path))[...,:3]


    res_list = []
    output = np.zeros((mask.shape[0], mask.shape[1], 3))
    for r in range(output.shape[0]):
        for c in range(output.shape[1]):
            if not (mask[r][c] < [20, 20, 20]).all():
                output[r][c] = origin[r][c]
            #     # print("xxxx")
            else:
                output[r][c] = res[r-ori_offset_r+offset_r][c-ori_offset_c+offset_c]
                res_list.append((r, c))
    Image.fromarray((output).astype(np.uint8)).save("test_ori.jpg")

    for rc in source:
        r, c = rc
        output[r+ori_offset_r][c+ori_offset_c] = origin[r+ori_offset_r][c+ori_offset_c]
    for rc in dest:
        r, c = rc
        output[r+ori_offset_r][c+ori_offset_c] = res[r+offset_r][c+offset_c]
        res_list.append((r+ori_offset_r, c+ori_offset_c))
    print(offset_r, offset_c)
    print(ori_offset_r, ori_offset_c)
    Image.fromarray((output).astype(np.uint8)).save("test.jpg")


    MatrixA = utils.calcMatrixA(res_list)
    b_r, b_g, b_b = utils.calcB(res_list, origin, res, ori_offset_r, ori_offset_c, offset_r, offset_c)

    x_r = jparse.solveMatrix(MatrixA, np.array(b_r))

    x_g = jparse.solveMatrix(MatrixA, np.array(b_g))

    x_b = jparse.solveMatrix(MatrixA, np.array(b_b))

    for i in range(res_list.__len__()):
        x, y = res_list[i]
        # print(x_r[i])
        if 0 <= x_r[i] <= 255:
            output[x][y][0] = x_r[i] 
        elif x_r[i] < 0:
            output[x][y][0] = 0
        else:
             output[x][y][0] = 255 
        if 0 <= x_g[i] <= 255:
            output[x][y][1] = x_g[i] 
        elif x_g[i] < 0:
            output[x][y][1] = 0
        else:
             output[x][y][1] = 255 
        if 0 <= x_b[i] <= 255:
            output[x][y][2] = x_b[i] 
        elif x_b[i] < 0:
            output[x][y][2] = 0
        else:
             output[x][y][2] = 255 
    Image.fromarray((output).astype(np.uint8)).save(output_path)

type=4
down_input_path = "../数据/down_input{}.jpg".format(type)
down_input_mask_path = "../数据/down_input{}_mask.jpg".format(type)
masked_path = "../数据/masked_input{}.png".format(type)
result_path = "../数据/input2/down_result_img002.jpg"
output_path = "../数据/output2/output002.jpg"
for i in range(1, 21):
    if i < 10:
        result_path = "../数据/input{}/down_result_img00{}.jpg".format(type, i)
        output_path = "../数据/output{}/output00{}.jpg".format(type, i)
    else:
        result_path = "../数据/input{}/down_result_img0{}.jpg".format(type, i)
        output_path = "../数据/output{}/output0{}.jpg".format(type, i)
    main(down_input_path, down_input_mask_path, masked_path, result_path, output_path, type)