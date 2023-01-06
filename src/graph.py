from collections import deque
import numpy as np
from PIL import Image
import copy
import cv2 as cv
import math
import networkx as nx
graph = nx.Graph()
# graph[(-1, -1)] = {} # patch A
# graph[(-2, -2)] = {} # patch B

def calcM(s_x, s_y, t_x, t_y, offset_r_a, offset_c_a, offset_r_b, offset_c_b, A, B):
    rgb_a_s = np.float32(A[s_x+offset_r_a][s_y+offset_c_a])
    rgb_a_t = np.float32(A[t_x+offset_r_a][t_y+offset_c_a])
    rgb_b_s = np.float32(B[s_x+offset_r_b][s_y+offset_c_b])
    rgb_b_t = np.float32(B[t_x+offset_r_b][t_y+offset_c_b])
    
    try:
        # print(rgb_a_s, rgb_a_t, rgb_b_s, rgb_b_t)
        ret = math.sqrt((rgb_a_s[0] - rgb_b_s[0])**2 + (rgb_a_s[1] - rgb_b_s[1])**2 + (rgb_a_s[2] - rgb_b_s[2])**2) + \
            math.sqrt((rgb_a_t[0] - rgb_b_t[0])**2 + (rgb_a_t[1] - rgb_b_t[1])**2 + (rgb_a_t[2] - rgb_b_t[2])**2) 
    except:
        print("xxxx")
    return ret

def isBlack(r, c, pic):
    return pic[r][c][0] < 5 and pic[r][c][1] < 5 and pic[r][c][2] < 5

def buildGraph(offset_r, offset_c, ori_offset_r, ori_offset_c, input_path, result_path, kernel, vertical=True):
    input_pic = np.asarray(Image.open(input_path))[...,:3]
    result_pic = np.asarray(Image.open(result_path))[...,:3]

    row = kernel.shape[0]
    col = kernel.shape[1]
    big_row = input_pic.shape[0]
    big_col = input_pic.shape[1]
    for r in range(row):
        for c in range(col):
            if isBlack(r, c, kernel):
                continue
            # graph[(r+offset_r, c+offset_c)] = {}
            graph.add_node((r, c))
            if vertical: # -1 -1 为原图， -2 -2 为新图
                if (r-1 >=0 and isBlack(r-1, c, kernel)) or r == 0:
                    # graph[(-1, -1)][(r+offset_r, c+offset_c)] = np.inf
                    graph.add_edge((-1, -1), (r, c), capacity=np.inf)
                elif (r + 1 < row and isBlack(r+1, c, kernel)) or r + 1 == row:
                    # graph[(-2, -2)][(r+offset_r, c+offset_c)] = np.inf
                    graph.add_edge((-2, -2), (r, c), capacity=np.inf)
            else:
                if (c-1 >= 0 and isBlack(r, c-1, kernel)) or c == 0:
                    graph.add_edge((-1, -1), (r, c), capacity=np.inf)
                    # graph[(-1, -1)][(r+offset_r, c+offset_c)] = np.inf
                elif (c+1 < col and isBlack(r, c+1, kernel)) or c + 1 == col:
                    # graph[(-2, -2)][(r+offset_r, c+offset_c)] = np.inf
                    graph.add_edge((-2, -2), (r, c), capacity=np.inf)
            
    for key in graph:
        if key == (-1, -1) or key == (-2, -2):
            continue
        r, c = key
        if (r+1, c) in graph:
            # graph[(r, c)][(r+1, c)] = calcM(r, c, r+1, c, input_pic, result_pic)
            graph.add_edge((r, c), (r+1, c), capacity=calcM(r, c, r+1, c, ori_offset_r, ori_offset_c, offset_r, offset_c, input_pic, result_pic))
        if (r-1, c) in graph:
            # graph[(r, c)][(r-1, c)] = calcM(r, c, r-1, c, input_pic, result_pic)
            graph.add_edge((r, c), (r-1, c), capacity=calcM(r, c, r-1, c, ori_offset_r, ori_offset_c, offset_r, offset_c, input_pic, result_pic))
        if (r, c+1) in graph:
            # graph[(r, c)][(r, c+1)] = calcM(r, c, r, c+1, input_pic, result_pic)
            graph.add_edge((r, c), (r, c+1), capacity=calcM(r, c, r, c+1, ori_offset_r, ori_offset_c, offset_r, offset_c, input_pic, result_pic))

        if (r, c-1) in graph:
            # graph[(r, c)][(r, c-1)] = calcM(r, c, r, c-1, input_pic, result_pic)
            graph.add_edge((r, c), (r, c-1), capacity=calcM(r, c, r, c-1, ori_offset_r, ori_offset_c, offset_r, offset_c, input_pic, result_pic))
    cut_value, partition = nx.minimum_cut(graph, (-1, -1), (-2, -2))
    reachable, unreachable = partition
    print(reachable.__len__(), unreachable.__len__())
    return partition
        
            