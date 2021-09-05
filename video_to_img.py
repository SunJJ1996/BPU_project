# encoding: utf-8

import cv2
import numpy as np
import os
import sys
import string


# 输出图片格式jpg
def save_image(image, addr, num):
    address = addr + str(num) + '.jpg'
    cv2.imwrite(address, image)


path = '/Users/sunjunjiao/Documents/Beam_pumping_unit_data/video/BPUvideo200723'
f_list = []
output = '/Users/sunjunjiao/Documents/Beam_pumping_unit_data/video/image_cut'
if not os.path.exists(output):
    os.makedirs(output)
for root, dirs, fs in os.walk(path):
    for f in fs:
        f_list.append(os.path.join(root, f))
print(root)
print(f_list)

for f in f_list:
    if str.find(f, '.py') != -1:
        print('.py file,skip!')
        continue
    if str.find(f, '.DS_Store') != -1:
        print('.DS_Store file,skip!')
        continue
    videoCapture = cv2.VideoCapture(f)
    success, frame = videoCapture.read()
    i = 0
    # k is pic number
    k = 0
    while success:
        i = i + 1
        # 隔23帧取一帧
        if i % 23 == 0:
            j = int(i / 5)
            k += 1
            # 输入视频格式m4v
            f = f.replace('/Users/sunjunjiao/Documents/Beam_pumping_unit_data/video/BPUvideo200723/', '')
            save_dir = os.path.join(output, f.replace('.mp4', ''))
            # print(save_dir)
            save_image(frame, save_dir + '_', k)
            if success:
                print('save image:', j)
        success, frame = videoCapture.read()
