import numpy as np
from PIL import Image
import utils
import jittor as jt
from graph import buildGraph

subdir = "../数据/input/"
dir_num = 1

pic_name_pref = "result_img"
pic_num = 1

# src = np.asarray(Image.open(filename))[...,:3]
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
    kernel, ori_offset_r, ori_offset_c = utils.bfs_get_kernel(maskedpath)
    js = jt.array(A)
    jk = jt.array(kernel)
    jy = utils.calcConvJittor(js, jk)
    return jy, kernel, ori_offset_r, ori_offset_c







read_and_mask("../数据/down_input1.jpg", "../数据/down_input1_mask.jpg", "./down_masked.png")
# utils.bfs_get_kernel("./masked.png")
y, kernel, ori_offset_r, ori_offset_c = calcConv("../数据/input1/down_result_img001.jpg", "./down_masked.png")
print(y.shape)

y_row = y.shape[0]
y_col = y.shape[1]
min = 10000000000
offset_r = -1
offset_c = -1
for r in range(y_row):
    for c in range(y_col):
        # print(y[r][c])
        if y[r][c] < min:
            min = y[r][c]
            offset_r = r
            offset_c = c


partition = buildGraph(offset_r, offset_c, ori_offset_r, ori_offset_c, "../数据/down_input1.jpg" , "../数据/input1/down_result_img001.jpg", kernel)

source, dest = partition
test = np.zeros((kernel.shape[0], kernel.shape[1], 3))

for xy in source:
    x, y = xy
    test[x][y] = kernel[x][y]
for xy in dest:
    x, y = xy
    test[x][y] = kernel[x][y]
Image.fromarray((test).astype(np.uint8)).save("test_graph.jpg")

origin = np.asarray(Image.open("../数据/down_input1.jpg"))[...,:3]
mask = np.asarray(Image.open("./down_masked.png"))[...,:3]
res = np.asarray(Image.open("../数据/input1/down_result_img001.jpg"))[...,:3]

output = np.zeros((mask.shape[0], mask.shape[1], 3))
for r in range(output.shape[0]):
    for c in range(output.shape[1]):
        if (mask[r][c] > [20, 20, 20]).all():
            output[r][c] = origin[r][c]
        #     # print("xxxx")
        else:
            output[r][c] = res[r-ori_offset_r+offset_r][c-ori_offset_c+offset_c]
Image.fromarray((output).astype(np.uint8)).save("test_ori.jpg")

for rc in source:
    r, c = rc
    output[r+ori_offset_r][c+ori_offset_c] = origin[r+ori_offset_r][c+ori_offset_c]
for rc in dest:
    r, c = rc
    output[r+ori_offset_r][c+ori_offset_c] = res[r+offset_r][c+offset_c]
    print("test")
print(offset_r, offset_c)
print(ori_offset_r, ori_offset_c)
Image.fromarray((output).astype(np.uint8)).save("test.jpg")

    
