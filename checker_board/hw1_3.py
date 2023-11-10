import math

import cv2 as cv
import sys

import numpy as np

def _polar_to_cartesian(lines):
    """
    극좌표 형식(x*cos_t + y*sin_t = rho)을
    데카르트 좌표평면 형식으로 변환
    (y = a*x + b)
    """
    cartesian = []
    for i in range(lines.shape[0]):
        rho, theta = lines[i][0]
        i = 1e-9
        cos_t = math.cos(theta)
        sin_t = math.sin(theta) + i
        a, b = -cos_t / sin_t, rho / sin_t  # y = ax + b
        cartesian.append((a, b))

    return np.array(cartesian)

def intersection_point(line1, line2, threshold=0.2):
    """
    두 직선의 교점을 구하는 함수
    두 직선은 y = ax + b 꼴이어야 함
    """
    a1, b1 = line1
    a2, b2 = line2
    if abs(a1 - a2) < threshold:
        return -1, -1
    x0 = -(b2 - b1) / (a2 - a1)
    y0 = a1 * x0 + b1
    return x0, y0

def draw_lines_polar(src, lines, color):
    if lines is None:
        print("lines is none!")
        exit(0)
    if src.ndim == 2:  # grayscale 이면 BGR로 변환
        dst = cv.cvtColor(src, cv.COLOR_GRAY2BGR)
    else:
        dst = src.copy()

    for i in range(lines.shape[0]):
        rho, theta = lines[i][0]
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        x0, y0 = rho * cos_t, rho * sin_t
        alpha = 1000
        pt1 = (int(x0 - alpha * sin_t), int(y0 + alpha * cos_t))
        pt2 = (int(x0 + alpha * sin_t), int(y0 - alpha * cos_t))
        cv.line(dst, pt1, pt2, color, 1, cv.LINE_AA)
    return dst

def main():
    filename = "board3.jpg" if len(sys.argv) < 2 else sys.argv[1]
    src = cv.imread(filename, cv.IMREAD_GRAYSCALE)

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

    # 가장 간격이 큰 두 직선 가로선 구하기
    max_h_line = h_lines[0][0]
    min_h_line = h_lines[0][0]
    for l in h_lines:
        rho, _ = l[0]
        max_rho, _ = max_h_line
        min_rho, _ = min_h_line
        if max_rho < rho:
            max_h_line = l[0]
        if rho < min_rho:
            min_h_line = l[0]
    # 세로선 구하기
    max_v_line = v_lines[0][0]
    min_v_line = v_lines[0][0]

    for l in v_lines:
        rho, theta = l[0]

        max_rho, mx_theta = max_v_line
        min_rho, mn_theta = min_v_line

        # 각도가 90도 넘어가면 rho의 부호가 바뀔 수 있음
        rho *= -1 if theta > math.pi / 2 else 1
        max_rho *= -1 if mx_theta > math.pi / 2 else 1
        min_rho *= -1 if mn_theta > math.pi / 2 else 1
        if max_rho < rho:
            max_v_line = l[0]
        if rho < min_rho:
            min_v_line = l[0]

    h_edges, v_edges = np.array([[min_h_line], [max_h_line]]), np.array([[min_v_line], [max_v_line]])

    edge = src.copy()
    # edge = draw_lines_polar(edge, h_lines, (0, 0, 255))
    # edge = draw_lines_polar(edge, v_lines, (255, 255, 0))
    edge = draw_lines_polar(edge, h_edges, (0, 0, 255))
    edge = draw_lines_polar(edge, v_edges, (0, 255, 0))
    edge = draw_lines_polar(edge, np.array([[[0, horizontal_line]]]), (255, 0, 0))

    # 모서리끼리 교점 계산
    h_edges_ab = _polar_to_cartesian(h_edges)
    v_edges_ab = _polar_to_cartesian(v_edges)
    points = []
    for h in h_edges_ab:
        for v in v_edges_ab:
            x, y = intersection_point(h, v)
            points.append((int(x), int(y)))
    for p in points:
        cv.circle(edge, p, 5, (0, 0, 255), -1)

    cv.imshow("edge", edge)

    # 꼭짓점 시계방향으로 정렬
    # 화면 기준으로는 clockwise=False여야 시계방향
    sorted_points = cv.convexHull(np.array(points), clockwise=False).astype(np.float32)

    w = 400
    dst_pts = np.array([[[0, 0]],
                        [[w - 1, 0]],
                        [[w - 1, w - 1]],
                        [[0, w - 1]]]).astype(np.float32)
    pers_mat = cv.getPerspectiveTransform(sorted_points, dst_pts)
    transformed_src = cv.warpPerspective(src, pers_mat, (w, w))
    cv.imshow("transformed", transformed_src)

    cv.waitKey()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()