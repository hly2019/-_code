import cv2

down_input1 = cv2.imread("./data/input1.jpg")

down_input1_masked = cv2.imread("./data/input1_mask.jpg")

# down_test = cv2.imread("../数据/input1/result_img001.jpg")

down_input1 = cv2.pyrDown(down_input1)
down_input1_masked = cv2.pyrDown(down_input1_masked)
# down_test = cv2.pyrDown(down_test)

down_input1 = cv2.pyrDown(down_input1)
down_input1_masked = cv2.pyrDown(down_input1_masked)
# down_test = cv2.pyrDown(down_test)

cv2.imwrite("./data/down_input1.jpg", down_input1)

cv2.imwrite("./data/down_input1_mask.jpg", down_input1_masked)

# cv2.imwrite("../数据/input1/down_result_img001.jpg", down_test)

# # cv2.waitKey()
# # cv2.destroyAllWindows()
for i in range(1, 20):
    if i < 10:
        filename = "result_img00{}.jpg".format(i)
    else:
        filename = "result_img0{}.jpg".format(i)
    print(filename)
    result = cv2.imread("./data/input1/{}".format(filename))
    result = cv2.pyrDown(result)
    result = cv2.pyrDown(result)
    cv2.imwrite("./data/down_input1/{}".format(filename), result)

