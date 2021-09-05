from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import matplotlib.pyplot as plt
from numpy.linalg import det

global Judgment_Processing


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0], \
                     detection[2][1], \
                     detection[2][2], \
                     detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        if str(detection[0], encoding="utf8") == 'Crank':
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
            cv2.putText(img,
                        detection[0].decode() +
                        " [" + str(round(detection[1] * 100, 2)) + "]",
                        (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [0, 255, 0], 2)
        # cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        # cv2.putText(img,
        #             detection[0].decode() +
        #             " [" + str(round(detection[1] * 100, 2)) + "]",
        #             (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             [0, 255, 0], 2)
    return img


def cvDrawPoint(detections, centerCircle_All, MeansCenter, img):
    for detection in detections:
        x, y, w, h = detection[2][0], \
                     detection[2][1], \
                     detection[2][2], \
                     detection[2][3]
        center = (int(x), int(y))
        center_size = 3
        if str(detection[0], encoding="utf8") != 'Beam_pumping_unit':
            cv2.circle(img, center, center_size, (0, 255, 0), -1)
    for centerCircle in centerCircle_All:
        if len(centerCircle):
            centerCir = (int(centerCircle[0]), int(centerCircle[1]))
            cv2.circle(img, centerCir, center_size, (255, 0, 0), -1)
    if len(MeansCenter):
        cv2.circle(img, MeansCenter, 4, (0, 0, 255), -1)
    return img


def findCircleCenter(Center_deList):
    p_list = []
    for detection in Center_deList:
        x, y, = detection[2][0], detection[2][1]
        p_list.append([int(x), int(y)])
    p0 = p_list[0]
    p1 = p_list[1]
    p2 = p_list[2]
    print(p0)
    print(p1)
    print(p2)
    center = points2circle(p0, p1, p2)
    print(center)
    return center


def findMeansCenter(center_list_All_use):
    Means_x = 0
    Means_y = 0
    for temp_center in center_list_All_use:
        Means_x = Means_x + temp_center[0]
        Means_y = Means_y + temp_center[1]
    if len(center_list_All_use):
        x = Means_x / len(center_list_All_use)
        y = Means_y / len(center_list_All_use)
        MeansCenter = (int(x), int(y))
        return MeansCenter


def point_center(p1, p2, p3):
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    x3, y3 = p3[0], p3[1]

    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    num1 = len(p1)
    num2 = len(p2)
    num3 = len(p3)

    k1 = 0
    k2 = 0
    # 输入检查
    if (num1 == num2) and (num2 == num3):
        if num1 == 2:
            p1 = np.append(p1, 0)
            p2 = np.append(p2, 0)
            p3 = np.append(p3, 0)
        elif num1 != 3:
            print('\t仅支持二维或三维坐标输入')
            return None
    else:
        print('\t输入坐标的维数不一致')
        return None

    # 共线检查
    temp01 = p1 - p2
    temp02 = p3 - p2
    temp03 = np.cross(temp01, temp02)
    temp = (temp03 @ temp03) / (temp01 @ temp01) / (temp02 @ temp02)
    if temp < 10 ** -6:
        print('\t三点共线, 无法确定圆')
        return None

    #########求三个点的两个弦
    # 求3弦的斜率
    if x1 != x2:
        k1 = (y2 - y1) / (x2 - x1)
    elif x1 == x2:
        k1 = 99999
    if x2 != x3:
        k2 = (y3 - y2) / (x3 - x2)
    elif x2 == x3:
        k2 = 99999
    if x1 != x3:
        k3 = (y3 - y1) / (x3 - x1)
    elif x1 == x3:
        k3 = 99999

    # 求两弦中点
    a1 = (x1 + x2) / 2
    b1 = (y1 + y2) / 2
    a2 = (x3 + x2) / 2
    b2 = (y3 + y2) / 2
    a3 = (x3 + x1) / 2
    b3 = (y3 + y1) / 2

    # 求两交线斜率
    if k1 != 0:
        C_k1 = -(1 / k1)
    elif k1 == 0:
        C_k1 = 99999
    if k2 != 0:
        C_k2 = -(1 / k2)
    elif k2 == 0:
        C_k2 = 99999
    if k3 != 0:
        C_k3 = -(1 / k3)
    elif k3 == 0:
        C_k3 = 99999

    # 求交点 one
    C_x1 = (C_k1 * a1 - C_k2 * a2 + b2 - b1) / (C_k1 - C_k2)
    C_y1 = C_k1 * (C_x1 - a1) + b1
    # 求交点 two
    C_x2 = (C_k1 * a1 - C_k3 * a3 + b3 - b1) / (C_k1 - C_k3)
    C_y2 = C_k1 * (C_x2 - a1) + b1
    # 求交点 three
    C_x3 = (C_k3 * a3 - C_k2 * a2 + b2 - b3) / (C_k3 - C_k2)
    C_y3 = C_k3 * (C_x3 - a3) + b3

    ####
    C_x = (C_x1 + C_x2 + C_x3) / 3
    C_y = (C_y1 + C_y2 + C_y3) / 3

    center = (int(C_x), int(C_y))
    return center


def points2circle(p1, p2, p3):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    num1 = len(p1)
    num2 = len(p2)
    num3 = len(p3)
    center = []
    # 输入检查
    if (num1 == num2) and (num2 == num3):
        if num1 == 2:
            p1 = np.append(p1, 0)
            p2 = np.append(p2, 0)
            p3 = np.append(p3, 0)
        elif num1 != 3:
            print('\t仅支持二维或三维坐标输入')
            return None
    else:
        print('\t输入坐标的维数不一致')
        return None

    # 共线检查
    temp01 = p1 - p2
    temp02 = p3 - p2
    temp03 = np.cross(temp01, temp02)
    temp = (temp03 @ temp03) / (temp01 @ temp01) / (temp02 @ temp02)
    if temp < 10 ** -6:
        print('\t三点共线, 无法确定圆')
        return None

    temp1 = np.vstack((p1, p2, p3))
    temp2 = np.ones(3).reshape(3, 1)
    mat1 = np.hstack((temp1, temp2))  # size = 3x4

    m = +det(mat1[:, 1:])
    n = -det(np.delete(mat1, 1, axis=1))
    p = +det(np.delete(mat1, 2, axis=1))
    q = -det(temp1)

    temp3 = np.array([p1 @ p1, p2 @ p2, p3 @ p3]).reshape(3, 1)
    temp4 = np.hstack((temp3, mat1))
    temp5 = np.array([2 * q, -m, -n, -p, 0])
    mat2 = np.vstack((temp4, temp5))  # size = 4x5

    A = +det(mat2[:, 1:])
    B = -det(np.delete(mat2, 1, axis=1))
    C = +det(np.delete(mat2, 2, axis=1))
    D = -det(np.delete(mat2, 3, axis=1))
    E = +det(mat2[:, :-1])

    pc = -np.array([B, C, D]) / 2 / A
    pc.astype(np.int)
    r = np.sqrt(B * B + C * C + D * D - 4 * A * E) / 2 / abs(A)
    r = int(r)
    center.append(int(pc[0]))
    center.append(int(pc[1]))
    return center


def cal_ang(point_1, point_2, point_3):
    """
    根据三点坐标计算夹角
    :param point_1: 点1坐标
    :param point_2: 点2坐标
    :param point_3: 点3坐标
    :return: 返回任意角的夹角值，这里只是返回点2的夹角
    """
    a = math.sqrt(
        (point_2[0] - point_3[0]) * (point_2[0] - point_3[0]) + (point_2[1] - point_3[1]) * (point_2[1] - point_3[1]))
    b = math.sqrt(
        (point_1[0] - point_3[0]) * (point_1[0] - point_3[0]) + (point_1[1] - point_3[1]) * (point_1[1] - point_3[1]))
    c = math.sqrt(
        (point_1[0] - point_2[0]) * (point_1[0] - point_2[0]) + (point_1[1] - point_2[1]) * (point_1[1] - point_2[1]))
    A = math.degrees(math.acos((a * a - b * b - c * c) / (-2 * b * c)))
    B = math.degrees(math.acos((b * b - a * a - c * c) / (-2 * a * c)))
    C = math.degrees(math.acos((c * c - a * a - b * b) / (-2 * a * b)))
    return B


def CountAngle(detectionD, detectionS, detectionB):
    ####DonkeyHead

    x, y, w, h = detectionD[2][0], \
                 detectionD[2][1], \
                 detectionD[2][2], \
                 detectionD[2][3]
    xmin, ymin, xmax, ymax = convertBack(
        float(x), float(y), float(w), float(h))
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    centerD = (int(x), int(y))
    ####SupportPoint
    x, y, w, h = detectionS[2][0], \
                 detectionS[2][1], \
                 detectionS[2][2], \
                 detectionS[2][3]
    xmin, ymin, xmax, ymax = convertBack(
        float(x), float(y), float(w), float(h))
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    centerS = (int(x), int(y))

    ####CenterBPU
    x, y, w, h = detectionB[2][0], \
                 detectionB[2][1], \
                 detectionB[2][2], \
                 detectionB[2][3]
    xmin, ymin, xmax, ymax = convertBack(
        float(x), float(y), float(w), float(h))
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    centerB = (int(x), int(y))
    Angle_DSB = cal_ang(centerD, centerS, centerB)
    return Angle_DSB


netMain = None
metaMain = None
altNames = None


def YOLO_BPU_Processing_Judgment(video_path, Judgement_thresh):
    global metaMain, netMain, altNames, detectionB
    configPath = "/data/sunjunjiao/yolo_darknet/darknet/cfg/yolov4-BPU200818_All_Dilated_diou_DenseNet_SAM.cfg"
    weightPath = "/data/sunjunjiao/yolo_darknet/darknet/backup/BPU200818WithCenter/yolov4-BPU200818_All_Dilated_diou_DenseNet_SAM_11000.weights"
    metaPath = "/data/sunjunjiao/yolo_darknet/darknet/data/BPU200818.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath) + "`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath) + "`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath) + "`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(video_path)
    Cap_W = cap.get(3)
    Cap_H = cap.get(4)
    cap.set(3, 1280)
    cap.set(4, 720)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # out = cv2.VideoWriter(
    #     "output_BPU.mp4", fourcc, 25.0,
    #     (darknet.network_width(netMain), darknet.network_height(netMain)), True)
    out = cv2.VideoWriter("output_BPU.mp4", fourcc, 25.0, (int(Cap_W), int(Cap_H)), True)
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                       darknet.network_height(netMain), 3)
    # Angle_list = []
    # time_list = []
    # Judge = []
    Judge_count = 0
    Center_determine_list = []
    center_list_All = []
    MeansCenter = []
    success, frame = cap.read()
    while success:
        # prev_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        ret, frame_read = cap.read()
        milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)
        seconds = milliseconds / 1000
        try:
            frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        except:
            break
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.75)
        ######### calculate the center of Crank
        # 判断10times

        Judge_count += 1

        if Judge_count >= 15:
            for detection in detections:
                temp_detec = str(detection[0], encoding="utf8")
                if temp_detec == 'Crank':
                    Center_determine_list.append(detection)
            Judge_count = 0
        if len(Center_determine_list) >= 3:
            Center_circle = findCircleCenter(Center_determine_list)
            center_list_All.append(Center_circle)
            MeansCenter = findMeansCenter(center_list_All)
            p1 = np.array(Center_circle)
            p2 = np.array(MeansCenter)
            p3 = p2 - p1
            p4 = math.hypot(p3[0], p3[1])
            if p4 >=40:
                del (center_list_All[-1])
                MeansCenter = findMeansCenter(center_list_All)
            del (Center_determine_list[0])
            del (Center_determine_list[0])

        #######
        #######
        # for detection in detections:
        #     temp_detec = str(detection[0], encoding="utf8")
        #     if temp_detec == 'Horse_head':
        #         detectionH = detection
        #     if temp_detec == 'Supporting_point':
        #         detectionS = detection
        #     if temp_detec == 'Beam_pumping_unit':
        #         detectionB = detection
        # try:
        #     Angle = CountAngle(detectionH, detectionS, detectionB)
        #     Angle_list.append(Angle)
        #     print(Angle)
        # except:
        #     continue

        print(seconds)
        print(center_list_All)

        # time_list.append(seconds)
        # Judge_count += 1
        # if Judge_count >= 10:
        #     Judge.append(Angle)
        #     Judge_count = 0
        #     if len(Judge) >= 10:
        #         del (Judge[0])
        #         if np.var(Judge) >= Judgement_thresh:
        #             Judgment_Processing = True
        #         else:
        #             Judgment_Processing = False
        #         print(Judgment_Processing)
        #         Judge = []

        image = cvDrawBoxes(detections, frame_resized)
        image = cvDrawPoint(detections, center_list_All, MeansCenter, image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (int(Cap_W), int(Cap_H)),
                           interpolation=cv2.INTER_LINEAR)
        out.write(image)
        # cv2.imshow('Demo', image)
        # cv2.waitKey(3)
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(time_list, Angle_list)
    # plt.show()
    out.release()
    cap.release()
    out.release()


if __name__ == "__main__":
    YOLO_BPU_Processing_Judgment("BPUvideo_19.mov", 20)
    # YOLO_BPU_Processing_Judgment("/data/sunjunjiao/BPUvideo/BPUvideo200718/Video_20200709122522.mp4", 20)
    # YOLO_BPU_Processing_Judgment("/data/sunjunjiao/BPUvideo/BPUstop200718/Video_20200709125251.mp4", 20)
