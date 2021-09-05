from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import matplotlib

matplotlib.use('Agg')
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
        if str(detection[0], encoding="utf8") != 'Center_Crank':
            cv2.rectangle(img, pt1, pt2, (255, 0, 0), 5)
            # cv2.putText(img,
            #             detection[0].decode() +
            #             " [" + str(round(detection[1] * 100, 2)) + "]",
            #             (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             [0, 255, 0], 2)
        # cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        # cv2.putText(img,
        #             detection[0].decode() +
        #             " [" + str(round(detection[1] * 100, 2)) + "]",
        #             (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             [0, 255, 0], 2)
    return img


def cvDrawPoint(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0], \
                     detection[2][1], \
                     detection[2][2], \
                     detection[2][3]
        center = (int(x), int(y))
        center_size = 3
        if str(detection[0], encoding="utf8") != 'Beam_pumping_unit':
            cv2.circle(img, center, center_size, (0, 255, 0), -1)
    coordinate_EndPoint = findCrankEndPoint(detections)
    cv2.circle(img, coordinate_EndPoint, center_size, (0, 255, 0), -1)
    return img


def Count_HorseHead_Deeping(detections):
    for detection in detections:
        # 驴头
        if str(detection[0], encoding="utf8") == 'Horse_head':
            Horse_head_x = detection[2][0]
            Horse_head_y = detection[2][1]
            Horse_head_point = (int(Horse_head_x), int(Horse_head_y))
        # 支点
        if str(detection[0], encoding="utf8") == 'Supporting_point':
            Supporting_point_x = detection[2][0]
            Supporting_point_y = detection[2][1]
            Supporting_point_point = (int(Supporting_point_x), int(Supporting_point_y))
    Location = (Horse_head_y - Supporting_point_y)
    return Location


def Count_All_Angle(detections):
    ## 将五个点的位置信息剥离出
    # 曲柄末端
    End_Crank_point = findCrankEndPoint(detections)

    for detection in detections:
        # 驴头
        if str(detection[0], encoding="utf8") == 'Horse_head':
            Horse_head_x = detection[2][0]
            Horse_head_y = detection[2][1]
            Horse_head_point = (int(Horse_head_x), int(Horse_head_y))
        # 支点
        if str(detection[0], encoding="utf8") == 'Supporting_point':
            Supporting_point_x = detection[2][0]
            Supporting_point_y = detection[2][1]
            Supporting_point_point = (int(Supporting_point_x), int(Supporting_point_y))
        # 后端
        if str(detection[0], encoding="utf8") == 'back_point':
            back_point_x = detection[2][0]
            back_point_y = detection[2][1]
            back_point_point = (int(back_point_x), int(back_point_y))
        # 减速器中心
        if str(detection[0], encoding="utf8") == 'Center_Crank':
            Center_Crank_x = detection[2][0]
            Center_Crank_y = detection[2][1]
            Center_Crank_point = (int(Center_Crank_x), int(Center_Crank_y))
    # 计算角1：H_S_C
    Angle_HSC = cal_ang(Horse_head_point, Supporting_point_point, Center_Crank_point)
    # 计算角2：H_S_E
    Angle_HSE = cal_ang(Horse_head_point, Supporting_point_point, End_Crank_point)
    # 计算角3：C_E_S
    Angle_CES = cal_ang(Center_Crank_point, End_Crank_point, Supporting_point_point)
    # 计算角4：E_C_S
    Angle_ECS = cal_ang(End_Crank_point, Center_Crank_point, Supporting_point_point)
    # 计算角5：B_C_E
    Angle_BCE = cal_ang(back_point_point, Center_Crank_point, End_Crank_point)
    # 计算角6：B_E_C
    Angle_BEC = cal_ang(back_point_point, End_Crank_point, Center_Crank_point)
    # 计算角7：H_B_C
    Angle_HBC = cal_ang(Horse_head_point, back_point_point, Center_Crank_point)
    # 计算角8：H_B_E
    Angle_HBE = cal_ang(Horse_head_point, back_point_point, End_Crank_point)
    return Angle_HSC, Angle_HSE, Angle_CES, Angle_ECS, Angle_BCE, Angle_BEC, Angle_HBC, Angle_HBE


def CountAngular_velocity(detections, n):
    ## 将五个点的位置信息剥离出
    # 曲柄末端
    End_Crank_point = findCrankEndPoint(detections)

    for detection in detections:
        # 驴头
        if str(detection[0], encoding="utf8") == 'Horse_head':
            Horse_head_x = detection[2][0]
            Horse_head_y = detection[2][1]
            Horse_head_point = (int(Horse_head_x), int(Horse_head_y))
        # 支点
        if str(detection[0], encoding="utf8") == 'Supporting_point':
            Supporting_point_x = detection[2][0]
            Supporting_point_y = detection[2][1]
            Supporting_point_point = (int(Supporting_point_x), int(Supporting_point_y))
        # 后端
        if str(detection[0], encoding="utf8") == 'back_point':
            back_point_x = detection[2][0]
            back_point_y = detection[2][1]
            back_point_point = (int(back_point_x), int(back_point_y))
        # 减速器中心
        if str(detection[0], encoding="utf8") == 'Center_Crank':
            Center_Crank_x = detection[2][0]
            Center_Crank_y = detection[2][1]
            Center_Crank_point = (int(Center_Crank_x), int(Center_Crank_y))

        # 计算公式中各个参数：w = (CE/BS)*(π*n/30)*(sin(BEC)/sin(HBE))
    CE = math.sqrt(math.pow(Center_Crank_x - End_Crank_point[0], 2) + math.pow(Center_Crank_y - End_Crank_point[1], 2))
    BS = math.sqrt(math.pow(back_point_x - Supporting_point_x, 2) + math.pow(back_point_y - Supporting_point_y, 2))
    Angle_HSC_get, Angle_HSE_get, Angle_CES_get, Angle_ECS_get, Angle_BCE_get, Angle_BEC_get, Angle_HBC_get, Angle_HBE_get = Count_All_Angle(
        detections)
    Angle_BEC_get_rad = (Angle_BEC_get / 180) * 3.14
    Angle_HBE_get_rad = (Angle_HBE_get / 180) * 3.14
    w = (CE / BS) * (3.14 * n / 30) * (math.sin(Angle_BEC_get_rad) / math.sin(Angle_HBE_get_rad))
    return w


# 平滑数据函数
def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re


def findCrankEndPoint(detections):
    global Crank_center_X
    for detection in detections:
        if str(detection[0], encoding="utf8") == 'Crank':
            Crank_X = detection[2][0];
            Crank_Y = detection[2][1]
        if str(detection[0], encoding="utf8") == 'Center_Crank':
            Crank_center_X = detection[2][0];
            Crank_center_Y = detection[2][1]
    CrankEnd_X = 2 * Crank_X - Crank_center_X
    CrankEnd_Y = 2 * Crank_Y - Crank_center_Y
    coordinate = (int(CrankEnd_X), int(CrankEnd_Y))
    return coordinate


def CountLenHS(detections):
    for detection in detections:
        if str(detection[0], encoding="utf8") == 'Horse_head':
            H_X = detection[2][0];
            H_Y = detection[2][1]
        if str(detection[0], encoding="utf8") == 'Supporting_point':
            S_X = detection[2][0];
            S_Y = detection[2][1]
    LenHS = math.sqrt(math.pow(H_X - S_X, 2) + math.pow(H_Y - S_Y, 2))
    return LenHS


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


def CountAngle(detectionUp, detectionMid, detectionDown):
    ####DonkeyHead

    x, y, w, h = detectionUp[2][0], \
                 detectionUp[2][1], \
                 detectionUp[2][2], \
                 detectionUp[2][3]
    xmin, ymin, xmax, ymax = convertBack(
        float(x), float(y), float(w), float(h))
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    centerD = (int(x), int(y))
    ####SupportPoint
    x, y, w, h = detectionMid[2][0], \
                 detectionMid[2][1], \
                 detectionMid[2][2], \
                 detectionMid[2][3]
    xmin, ymin, xmax, ymax = convertBack(
        float(x), float(y), float(w), float(h))
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    centerS = (int(x), int(y))

    ####CenterBPU
    x, y, w, h = detectionDown[2][0], \
                 detectionDown[2][1], \
                 detectionDown[2][2], \
                 detectionDown[2][3]
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


def YOLO_BPU_Processing_Judgment(video_path):
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
    out = cv2.VideoWriter("output_BPU_withCenter.mp4", fourcc, 25.0, (int(Cap_W), int(Cap_H)), True)
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                       darknet.network_height(netMain), 3)
    Angle_HSC_get_list = []
    Angle_HSE_get_list = []
    Angle_CES_get_list = []
    Angle_ECS_get_list = []
    Angle_BCE_get_list = []
    Angle_BEC_get_list = []
    Angle_HBC_get_list = []
    Angle_HBE_get_list = []
    time_list = []
    LenHS_list = []
    Horse_head_list = []
    Angular_velocity_list = []
    Angular_velocity_Instantaneous_center_list = []
    Velocity_list = []
    Velocity_Instantaneous_center_list = []
    Angular_acceleration_list = []
    Velocity_acceleration_list = []
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

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.5)
        print("time:", seconds)

        try:
            Angle_HSC_get, Angle_HSE_get, Angle_CES_get, Angle_ECS_get, Angle_BCE_get, Angle_BEC_get, Angle_HBC_get, Angle_HBE_get = Count_All_Angle(
                detections)
            Angle_HSC_get_list.append(Angle_HSC_get)
            Angle_HSE_get_list.append(Angle_HSE_get)
            Angle_CES_get_list.append(Angle_CES_get)
            Angle_ECS_get_list.append(Angle_ECS_get)
            Angle_BCE_get_list.append(Angle_BCE_get)
            Angle_BEC_get_list.append(Angle_BEC_get)
            Angle_HBC_get_list.append(Angle_HBC_get)
            Angle_HBE_get_list.append(Angle_HBE_get)
        except:
            continue
        try:
            LenHS_list.append(CountLenHS(detections))
        except:
            continue
        time_list.append(seconds)

        # 计算驴头位移
        try:
            Horse_head_location = Count_HorseHead_Deeping(detections)
            Horse_head_list.append(Horse_head_location)
        except:
            Horse_head_list.append(0)

        # 计算角速度
        # 数据法
        try:
            Angular_origin = Angle_HSC_get_list[-2]
            Angular_naw = Angle_HSC_get_list[-1]
            Det_Angle = abs(Angular_naw - Angular_origin)
            Det_t = abs(time_list[-1] - time_list[-2])
            Angular_velocity = Det_Angle / Det_t
            # 转换弧度
            Angular_velocity_rad = (Angular_velocity / 180) * 3.14
            Angular_velocity_list.append(Angular_velocity_rad)
            # print(Angular_origin)
            # print(Angular_naw)
            # print(Angular_velocity)
        except:
            Angular_velocity_list.append(0)

        # 公式法
        try:
            Angular_velocity_Instantaneous_center = CountAngular_velocity(detections, 4)
            Angular_velocity_Instantaneous_center_list.append(Angular_velocity_Instantaneous_center)
        except:
            Angular_velocity_Instantaneous_center_list.append(0)

        # 数据平滑
        Angular_velocity_list_smooth = moving_average(Angular_velocity_list, 80)

        # 计算速度
        # 数据法
        try:
            LenHS_real = CountLenHS(detections)
            Velocity = Angular_velocity_rad * LenHS_real * 3.14
            Velocity_list.append(Velocity)
        except:
            Velocity_list.append(0)

        # 公式法
        try:
            LenHS_real = CountLenHS(detections)
            Velocity_Instantaneous_center = Angular_velocity_Instantaneous_center * LenHS_real * 3.14
            Velocity_Instantaneous_center_list.append(Velocity_Instantaneous_center)
        except:
            Velocity_Instantaneous_center_list.append(0)

        # 数据平滑
        Velocity_list_smooth = moving_average(Velocity_list, 80)

        # 计算角加速度
        # 数据法
        try:
            Angular_Velocity_origin = Angular_velocity_Instantaneous_center_list[-2]
            Angular_Velocity_naw = Angular_velocity_Instantaneous_center_list[-1]
            Det_Angle_Velocity = (Angular_Velocity_origin - Angular_Velocity_naw)
            Det_t = abs(time_list[-2] - time_list[-1])
            Angular_acceleration = Det_Angle_Velocity / Det_t
            Angular_acceleration_list.append(Angular_acceleration)
        except:
            Angular_acceleration_list.append(0)

        # 数据平滑
        Angular_acceleration_list_smooth = moving_average(Angular_acceleration_list, 80)

        # 计算加速度
        # 数据法
        try:
            Velocity_origin = Velocity_Instantaneous_center_list[-2]
            Velocity_naw = Velocity_Instantaneous_center_list[-1]
            Det_Velocity = (Velocity_origin - Velocity_naw)
            Det_t = abs(time_list[-2] - time_list[-1])
            Velocity_Acceleration = Det_Velocity / Det_t
            Velocity_acceleration_list.append(Velocity_Acceleration)
        except:
            Velocity_acceleration_list.append(0)

        # 数据平滑
        Velocity_acceleration_list_smooth = moving_average(Velocity_acceleration_list, 80)

        image = cvDrawBoxes(detections, frame_resized)
        image = cvDrawPoint(detections, image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (int(Cap_W), int(Cap_H)),
                           interpolation=cv2.INTER_LINEAR)
        out.write(image)
        # cv2.imshow('Demo', image)
        cv2.waitKey(3)

    # # 打印所有角度的曲线
    # fig = plt.figure()
    # ax = fig.add_subplot(2, 4, 1)
    # ax.plot(time_list, Angle_HSC_get_list)
    # plt.xlabel('Time')
    # plt.ylabel('HSC/°')
    #
    # ax = fig.add_subplot(2, 4, 2)
    # ax.plot(time_list, Angle_HSE_get_list)
    # plt.xlabel('Time')
    # plt.ylabel('HSE/°')
    #
    # ax = fig.add_subplot(2, 4, 3)
    # ax.plot(time_list, Angle_CES_get_list)
    # plt.xlabel('Time')
    # plt.ylabel('CES/°')
    #
    # ax = fig.add_subplot(2, 4, 4)
    # ax.plot(time_list, Angle_ECS_get_list)
    # plt.xlabel('Time')
    # plt.ylabel('ECS/°')
    #
    # ax = fig.add_subplot(2, 4, 5)
    # ax.plot(time_list, Angle_BCE_get_list)
    # plt.xlabel('Time')
    # plt.ylabel('BCE/°')
    #
    # ax = fig.add_subplot(2, 4, 6)
    # ax.plot(time_list, Angle_BEC_get_list)
    # plt.xlabel('Time')
    # plt.ylabel('BEC/°')
    #
    # ax = fig.add_subplot(2, 4, 7)
    # ax.plot(time_list, Angle_HBC_get_list)
    # plt.xlabel('Time')
    # plt.ylabel('HBC/°')
    #
    # ax = fig.add_subplot(2, 4, 8)
    # ax.plot(time_list, Angle_HBE_get_list)
    # plt.xlabel('Time')
    # plt.ylabel('HBE/°')
    #
    # # plt.show()
    # plt.savefig("All_angle_BPU.jpg")

    # 打印所有角度的曲线
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(time_list, Angle_HSC_get_list)
    plt.xlabel('Time')
    plt.ylabel('HSC/°')
    plt.savefig("Angle_HSC.jpg")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(time_list, Angle_HSE_get_list)
    plt.xlabel('Time')
    plt.ylabel('HSE/°')
    plt.savefig("Angle_HSE.jpg")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(time_list, Angle_CES_get_list)
    plt.xlabel('Time')
    plt.ylabel('CES/°')
    plt.savefig("Angle_CES.jpg")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(time_list, Angle_ECS_get_list)
    plt.xlabel('Time')
    plt.ylabel('ECS/°')
    plt.savefig("Angle_ECS.jpg")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(time_list, Angle_BCE_get_list)
    plt.xlabel('Time')
    plt.ylabel('BCE/°')
    plt.savefig("Angle_BCE.jpg")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(time_list, Angle_BEC_get_list)
    plt.xlabel('Time')
    plt.ylabel('BEC/°')
    plt.savefig("Angle_BEC.jpg")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(time_list, Angle_HBC_get_list)
    plt.xlabel('Time')
    plt.ylabel('HBC/°')
    plt.savefig("Angle_HBC.jpg")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(time_list, Angle_HBE_get_list)
    plt.xlabel('Time')
    plt.ylabel('HBE/°')
    plt.savefig("Angle_HBE.jpg")

    # 打印位移曲线
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(time_list, Horse_head_list)
    plt.xlabel('Time')
    plt.ylabel('Horse_head_location(pixel)')
    plt.savefig("Horse_head_location_BPU.jpg")

    # 打印角速度曲线
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(time_list, Angular_velocity_list_smooth)
    plt.xlabel('Time')
    plt.ylabel('Angular_velocity(rad/s)')
    plt.savefig("Angular_velocity_BPU.jpg")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(time_list, Angular_velocity_Instantaneous_center_list)
    plt.xlabel('Time')
    plt.ylabel('Angular_velocity(rad/s)')
    plt.savefig("Angular_velocity_Instantaneous_center_BPU.jpg")

    # 打印速度曲线
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(time_list, Velocity_list_smooth)
    plt.xlabel('Time')
    plt.ylabel('Velocity(pixel/s)')
    plt.savefig("Velocity_BPU.jpg")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(time_list, Velocity_Instantaneous_center_list)
    plt.xlabel('Time')
    plt.ylabel('Velocity(pixel/s)')
    plt.savefig("Velocity__Instantaneous_center_BPU.jpg")

    # 打印角加速度曲线
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(time_list, Angular_acceleration_list_smooth)
    plt.xlabel('Time')
    plt.ylabel('Angular_acceleration(rad/s^2)')
    plt.savefig("Angular_acceleration_BPU.jpg")

    # 打印加速度曲线
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(time_list, Velocity_acceleration_list_smooth)
    plt.xlabel('Time')
    plt.ylabel('Velocity_acceleration(pixel/s^2)')
    plt.savefig("Velocity_acceleration_BPU.jpg")

    out.release()
    cap.release()
    out.release()


if __name__ == "__main__":
    # YOLO_BPU_Processing_Judgment("BPUvideo_19.mov")
    # YOLO_BPU_Processing_Judgment("BPUvideo_19_part.mov")
    # YOLO_BPU_Processing_Judgment("BPUvideo_7.mp4")
    YOLO_BPU_Processing_Judgment("BPUvideo_2.mp4")
    # YOLO_BPU_Processing_Judgment("/data/sunjunjiao/BPUvideo/BPUvideo200718/Video_20200709122522.mp4", 20)
    # YOLO_BPU_Processing_Judgment("/data/sunjunjiao/BPUvideo/BPUstop200718/Video_20200709125251.mp4", 20)
