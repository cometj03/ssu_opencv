import math

import cv2 as cv
import sys

import numpy as np

from util import draw_lines_polar, boards

def main():
    for b in boards:
        test(b)

def test(src):
    # filename = "checker5.jpg" if len(sys.argv) < 2 else sys.argv[1]
    # src = cv.imread(filename, cv.IMREAD_GRAYSCALE)

    if src is None:
        print("Image load failed!")
        exit()

    width = 700
    ratio = width / src.shape[1]
    inter_flag = cv.INTER_CUBIC if (ratio > 1) else cv.INTER_AREA
    src = cv.resize(src, (0, 0), fx=ratio, fy=ratio, interpolation=inter_flag)

    blur = cv.GaussianBlur(src, (0, 0), 1.2)
    # _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 71, 5)

    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 3))
    dilate = cv.morphologyEx(thresh, cv.MORPH_DILATE, horizontal_kernel, iterations=1)
    dilate = cv.morphologyEx(dilate, cv.MORPH_CLOSE, horizontal_kernel, iterations=2)
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 5))
    dilate = cv.morphologyEx(dilate, cv.MORPH_DILATE, vertical_kernel, iterations=1)
    dilate = cv.morphologyEx(dilate, cv.MORPH_CLOSE, vertical_kernel, iterations=2)

    cv.imshow("dilate", dilate)

    edge = cv.Canny(dilate, 100, 100)
    lines = cv.HoughLines(edge, 1, math.pi / 180, 100)

    # 가로 선과 세로 선 나누기
    horizontal_line = math.pi / 2
    thresh_angle = math.pi / 4
    h_lines, v_lines = [], []
    while len(h_lines) < 2 or len(v_lines) < 2:
        h_lines.clear()
        v_lines.clear()
        for line in lines:
            rho, theta = line[0]
            if abs(theta - horizontal_line) < thresh_angle:
                h_lines.append([line[0]])
            else:
                v_lines.append([line[0]])

        if (len(h_lines) < 2 or len(v_lines) < 2) and len(h_lines) > len(v_lines):
            horizontal_line += math.pi / 30  # 기준 선 조금씩 조정

    h_lines, v_lines = np.array(h_lines), np.array(v_lines)

    edge = src.copy()
    edge = draw_lines_polar(edge, h_lines, (0, 0, 255))
    edge = draw_lines_polar(edge, v_lines, (255, 255, 0))
    edge = draw_lines_polar(edge, np.array([[[100, horizontal_line]]]), (0, 255, 0))


    cv.imshow("edge", edge)

    cv.waitKey()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()