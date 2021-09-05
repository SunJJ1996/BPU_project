from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import matplotlib.pyplot as plt

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
        center = (int(x), int(y))
        center_size = 1
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
        cv2.circle(img, center, center_size, (0, 255, 0), 1)
    return img


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


def CountAngle(detectionH, detectionS, detectionB):
    ####DonkeyHead

    x, y, w, h = detectionH[2][0], \
                 detectionH[2][1], \
                 detectionH[2][2], \
                 detectionH[2][3]
    xmin, ymin, xmax, ymax = convertBack(
        float(x), float(y), float(w), float(h))
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    centerH = (int(x), int(y))
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
    Angle_HSB = cal_ang(centerH, centerS, centerB)
    return Angle_HSB


netMain = None
metaMain = None
altNames = None


def YOLO_BPU_Processing_Judgment(video_path, Judgement_thresh):
    global metaMain, netMain, altNames, detectionB
    configPath = "/data/sunjunjiao/yolo_darknet/darknet/cfg/yolov4-BPU200809_All_Dilated_diou_DenseNet_SAM.cfg"
    weightPath = "/data/sunjunjiao/yolo_darknet/darknet/backup/BPU200809AllProject/yolov4-BPU200809_All_Dilated_diou_DenseNet_SAM_best.weights"
    metaPath = "/data/sunjunjiao/yolo_darknet/darknet/data/BPU200809.data"
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
    cap.set(3, 1280)
    cap.set(4, 720)
    out = cv2.VideoWriter(
        "output_BPU.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
        (darknet.network_width(netMain), darknet.network_height(netMain)))
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                       darknet.network_height(netMain), 3)
    Angle_list = []
    time_list = []
    Judge = []
    Judge_count = 0
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
        #######
        for detection in detections:
            temp_detec = str(detection[0], encoding="utf8")
            if temp_detec == 'Horse_head':
                detectionD = detection
            if temp_detec == 'Supporting_point':
                detectionS = detection
            if temp_detec == 'Beam_pumping_unit':
                detectionB = detection
        try:
            Angle = CountAngle(detectionD, detectionS, detectionB)
            Angle_list.append(Angle)
            print(Angle)
        except:
            continue

        print(seconds)

        time_list.append(seconds)
        Judge_count += 1
        if Judge_count >= 10:
            Judge.append(Angle)
            Judge_count = 0
            if len(Judge) >= 10:
                del (Judge[0])
                if np.var(Judge) >= Judgement_thresh:
                    Judgment_Processing = True
                else:
                    Judgment_Processing = False
                print(Judgment_Processing)
                Judge = []
                # print(Judeg)
                # print(np.var(Judeg))

        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(1/(time.time()-prev_time))
        cv2.imshow('Demo', image)
        cv2.waitKey(3)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(time_list, Angle_list)
    plt.show()
    cap.release()
    out.release()


if __name__ == "__main__":
    YOLO_BPU_Processing_Judgment("BPUvideo_2.mp4", 20)
    # YOLO_BPU_Processing_Judgment("/data/sunjunjiao/BPUvideo/BPUvideo200718/Video_20200709122522.mp4", 20)
    # YOLO_BPU_Processing_Judgment("/data/sunjunjiao/BPUvideo/BPUstop200718/Video_20200709125251.mp4", 20)

