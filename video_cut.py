import os

import cv2
import numpy as np

from PIL import Image





# 使用opencv按一定间隔截取视频帧，并保存为图片

vc = cv2.VideoCapture('/Users/sunjunjiao/Documents/Beam_pumping_unit_data/test_video/output_red_2.mp4')  # 读取视频文件
# vc = cv2.VideoCapture('/Users/sunjunjiao/Documents/Beam_pumping_unit_data/video/BPUvideo_2.mp4')  # 读取视频文件

c = 0
print("------------")
if vc.isOpened():  # 判断是否正常打开
    print("yes")
    rval, frame = vc.read()
else:
    rval = False
    print("false")

timeF = 33  # 视频帧计数间隔频率

while rval:  # 循环读取视频帧
    rval, frame = vc.read()
    print(c, timeF, c % timeF)
    if c == 0:
        image = frame
    if c % timeF == 0 and c != 0:  # 每隔timeF帧进行存储操作
        print("write...")
        try:
            image = np.hstack((image, frame))
        except:
            continue
        # cv2.imwrite('/Users/sunjunjiao/Documents/Beam_pumping_unit_data/video_cut/origin_' + str(c) + '.jpg',
        #             frame)  # 存储为图像
        print("success!")
    c = c + 1
cv2.imwrite('/Users/sunjunjiao/Documents/Beam_pumping_unit_data/video_cut/output_red.jpg', image)  # 存储为图像
# cv2.imwrite('/Users/sunjunjiao/Documents/Beam_pumping_unit_data/video_cut/origin.jpg', image)  # 存储为图像
cv2.waitKey(1)
vc.release()
print("==================================")
