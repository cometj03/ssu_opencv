import cv2 as cv
import sys

import numpy as np

click_cnt = 0
src_points = np.zeros((4, 2), dtype=np.float32)

def main():
    filename = "checker5.jpg" if len(sys.argv) < 2 else sys.argv[1]
    src = cv.imread(filename, cv.IMREAD_COLOR)

    width = 700
    ratio = width / src.shape[1]
    inter_flag = cv.INTER_CUBIC if (ratio > 1) else cv.INTER_AREA
    src = cv.resize(src, (0, 0), fx=ratio, fy=ratio, interpolation=inter_flag)
    src_circle = src.copy()

    if src is None:
        print("Image load failed!")
        exit()

    def on_mouse(event, x, y, flags, param):
        global click_cnt, src_points
        if event != cv.EVENT_LBUTTONDOWN:
            return
        if click_cnt < 4:
            src_points[click_cnt, :] = np.array([x, y]).astype(np.float32)
            click_cnt += 1

            cv.circle(src_circle, (x, y), 5, (0, 0, 255), -1)
            cv.imshow("src", src_circle)
        if click_cnt == 4:
            click_cnt += 1

            # 화면 기준으로는 clockwise=False여야 시계방향
            points = cv.convexHull(src_points, clockwise=False)

            if len(points) != 4:
                print("점을 볼록 다각형으로 찍어주세요!")
                exit()

            # todo: 왼쪽 위의 점을 맨 처음으로 옮기기
            w = 400
            dst_pts = np.array([[0, 0],
                                [w - 1, 0],
                                [w - 1, w - 1],
                                [0, w - 1]]).astype(np.float32)

            pers_mat = cv.getPerspectiveTransform(src_points, dst_pts)
            dst = cv.warpPerspective(src, pers_mat, (w, w))
            cv.imshow("dst", dst)


    cv.namedWindow("src")
    cv.setMouseCallback("src", on_mouse)
    cv.imshow("src", src)

    cv.waitKey()
    cv.destroyAllWindows()



if __name__ == "__main__":
    main()