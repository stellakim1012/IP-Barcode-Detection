from __future__ import print_function
import argparse
import numpy as np
import cv2
import os
import glob
import time


def detectBarcode(img):
    # 수평 및 수직 방향으로 경계 강도 계산
    sharpx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = -1)
    sharpx = cv2.convertScaleAbs(sharpx)
    sharpy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = -1)
    sharpy = cv2.convertScaleAbs(sharpy)

    # 수평 방향 기준 바코드 후보 영역 검출
    dstx = cv2.subtract(sharpx, sharpy)
    dstx = cv2.GaussianBlur(dstx, (9, 7), 0)
    th, dstx = cv2.threshold(dstx, 100, 200, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (61, 9))
    dstx = cv2.morphologyEx(dstx, cv2.MORPH_CLOSE, kernel)
    dstx = cv2.erode(dstx, kernel, iterations=3)
    dstx = cv2.dilate(dstx, kernel, iterations=3)

    (contours, hierarchy) = cv2.findContours(dstx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        maxx = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
    else:
        maxx = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # 수직 방향 기준 바코드 후보 영역 검출
    dsty = cv2.subtract(sharpy, sharpx)
    dsty = cv2.GaussianBlur(dsty, (7, 9), 0)
    th, dsty = cv2.threshold(dsty, 100, 200, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 61))
    dsty = cv2.morphologyEx(dsty, cv2.MORPH_CLOSE, kernel)
    dsty = cv2.erode(dsty, kernel, iterations=3)
    dsty = cv2.dilate(dsty, kernel, iterations=3)

    (contours, hierarchy) = cv2.findContours(dsty, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        maxy = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
    else:
        maxy = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # 수평 및 수직 방향 연결 요소 크기 비교
    if (len(maxx) > len(maxy)):
        rect = cv2.minAreaRect(maxx)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        if (box[2][0] > box[0][0]):
            temp = box[2][0]
            box[2][0] = box[0][0]
            box[0][0] = temp
    else:
        rect = cv2.minAreaRect(maxy)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        if (box[2][0] > box[0][0]):
            temp = box[2][0]
            box[2][0] = box[0][0]
            box[0][0] = temp
    return box


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to the dataset folder")
    ap.add_argument("-r", "--detectset", required=True, help="path to the detectset folder")
    ap.add_argument("-f", "--detect", required=True, help="path to the detect file")
    args = vars(ap.parse_args())

    dataset = args["dataset"]
    detectset = args["detectset"]
    detectfile = args["detect"]

    # 결과 영상 저장 폴더 존재 여부 확인
    if (not os.path.isdir(detectset)):
        os.mkdir(detectset)

    # 결과 영상 표시 여부
    verbose = False

    # 검출 결과 위치 저장을 위한 파일 생성
    f = open(detectfile, "wt", encoding="UTF-8")  # UT-8로 인코딩

    timestart = time.time()
    # 바코드 영상에 대한 바코드 영역 검출
    for imagePath in glob.glob(dataset + "/*.jpg"):
        print(imagePath, '처리중...')

        # 영상을 불러오고 그레이 스케일 영상으로 변환
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 바코드 검출
        points = detectBarcode(gray)

        # 바코드 영역 표시
        detectimg = cv2.rectangle(image, (points[2][0], points[2][1]), (points[0][0], points[0][1]), (0, 255, 0), 2)  # 이미지에 사각형 그리기

        # 결과 영상 저장
        loc1 = imagePath.rfind("\\")
        loc2 = imagePath.rfind(".")
        fname = 'result/' + imagePath[loc1 + 1: loc2] + '_res.jpg'
        cv2.imwrite(fname, detectimg)

        # 검출한 결과 위치 저장
        f.write(imagePath[loc1 + 1: loc2])
        f.write("\t")
        f.write(str(points[2][0]))
        f.write("\t")
        f.write(str(points[2][1]))
        f.write("\t")
        f.write(str(points[0][0]))
        f.write("\t")
        f.write(str(points[0][1]))
        f.write("\n")

        if verbose:
            cv2.imshow("image", image)
            cv2.waitKey(0)

    print("time : ", time.time() - timestart)