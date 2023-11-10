import cv2 as cv
import numpy as np
import math

def _resize(src, width=500):
    ratio = width / src.shape[1]
    # 영상 확대 시 CUBIC, 축소 시 AREA
    inter_flag = cv.INTER_CUBIC if (ratio > 1) else cv.INTER_AREA
    new = cv.resize(src, (0, 0), fx=ratio, fy=ratio, interpolation=inter_flag)
    return new

def imread(filename, width=700):
    src = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    # src = _resize(src, width)
    return src

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
    a1, b1 = line1
    a2, b2 = line2
    if abs(a1 - a2) < threshold:
        return -1, -1
    x0 = -(b2 - b1) / (a2 - a1)
    y0 = a1 * x0 + b1
    return x0, y0

def intersection_points(lines, width, height):
    # y = ax + b
    lines = _polar_to_cartesian(lines)
    points = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            x, y = intersection_point(lines[i], lines[j])
            if 0 <= x <= width and 0 <= y <= height:
                points.append((int(x), int(y)))
    return np.array(points)

def rect(src):

    pass

def sharpen(src):
    blurred = cv.GaussianBlur(src, (0, 0), 1.0)
    alpha = 1.0
    dst = cv.addWeighted(src, 1 + alpha, blurred, -alpha, 0)
    return dst

def detect_edge(src, thres1, thres2):
    edge = cv.Canny(src, thres1, thres2)
    edge = cv.dilate(edge, None)
    edge = cv.erode(edge, None)

    return edge

def detect_line(edge, threshold, thres_angle=math.pi/6):
    horizontal, vertical = [], []
    lines = cv.HoughLines(edge, 1, math.pi / 180, threshold)

    # 세로 직선의 각도가 0이고, 시계방향으로 커짐. 최대는 pi
    for i in range(lines.shape[0]):
        rho, theta = lines[i][0]
        if theta < thres_angle or math.pi - theta < thres_angle:
            vertical.append([lines[i][0]])
        elif abs(theta - math.pi / 2) < thres_angle:
            horizontal.append([lines[i][0]])

    h, v = np.array(horizontal), np.array(vertical)
    return h, v

def detect_contours(edge, min_area=400):
    contours, _ = cv.findContours(edge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    large_contours = []
    for contour in contours:
        if cv.contourArea(contour) > min_area:
            large_contours.append(contour)

    return large_contours

def calculate_size(src):
    edge = cv.Canny(src, 100, 180)
    edge = cv.dilate(edge, None)
    edge = cv.erode(edge, None)
    lines = cv.HoughLines(edge, 1, math.pi / 180, 200)

    points = intersection_points(lines, src.shape[1], src.shape[0])

    # draw point
    dst = cv.cvtColor(src, cv.COLOR_GRAY2BGR)
    for p in points:
        cv.circle(dst, (p[0], p[1]), 5, (0, 255, 0), -1)

    dst = draw_lines_polar(dst, lines, (0, 0, 255))
    cv.imshow('edge', edge)
    cv.imshow('measure_size', dst)

    return 0, 0


# def test(board):
#     board = cv.GaussianBlur(board, (0, 0), 1)
#     # board = contrast(board)
#
#     edge = cv.Canny(board, 130, 160)
#     edge = cv.dilate(edge, None)
#     edge = cv.erode(edge, None)
#
#     # todo: transform 과정 추가
#
#     lines = cv.HoughLines(edge, 1, math.pi / 180, 300)  # 픽셀 단위, 각도 단위, threshold
#     dst = cv.cvtColor(edge, cv.COLOR_GRAY2BGR)
#     dst = draw_lines_polar(dst, lines)
#
#     contours, hierarchy = cv.findContours(edge, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
#
#     # idx = 0
#     # while idx >= 0:
#     #     c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
#     #     cv.drawContours(dst, contours, idx, c, -1, cv.LINE_8, hierarchy)
#     #     idx = hierarchy[0, idx, 0]
#     #     print(idx)
#     #     cv.imshow("line", dst)
#     #     cv.waitKey()
#
#     for contour in contours:
#         epsilon = cv.arcLength(contour, True) * 0.02
#         points = cv.approxPolyDP(contour, epsilon, True)
#         points = cv.convexHull(points)
#
#         # if len(points) != 4:
#         #     # 검출 안 됐을 때 한 번 더
#         #     points = cv.approxPolyDP(points, epsilon * 10, True)
#
#         if len(points) == 4:
#             for i in range(len(points)):
#                 cv.line(dst, points[i][0], points[i-1][0], (0, 255, 0), 1)
#
#     cv.imshow("origin", board)
#     cv.imshow("edge", edge)
#     cv.imshow("line", dst)
#
#     cv.waitKey()

boards = [
    # imread("checker1_full.png"),
    # imread("checker3_full_with_pieces.png"),
    imread("board1.jpg"),
    imread("board2.jpg"),
    imread("board3.jpg"),
    imread("checker6.jpg"),
    imread("checker7_with_pieces.jpg"),
    imread("checker8_with_pieces.jpg"),
]
def main():

    for b in boards[:]:
        # test(b)
        pass
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
