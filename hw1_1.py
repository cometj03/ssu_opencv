import cv2

from main import *

def _preprocessing(src):
    dst = cv.GaussianBlur(src, (0, 0), 1)
    dst = sharpen(dst)
    return dst

def main():
    for board in boards[:]:
        src = _preprocessing(board)
        dst = cv.cvtColor(src, cv.COLOR_GRAY2BGR)

        edge = detect_edge(board, 150, 180)
        # edge = detect_edge(board, 120, 150)

        # h_line, v_line = detect_line(edge, 120)
        #
        # dst = draw_lines_polar(dst, h_line, (255, 255, 0))
        # dst = draw_lines_polar(dst, v_line, (0, 0, 255))

        contours = detect_contours(edge)
        for cnt in contours:
            epsilon = cv.arcLength(cnt, True) * 0.01
            approx = cv.approxPolyDP(cnt, epsilon, True)
            approx = cv.convexHull(approx)

            cv.drawContours(dst, [approx], 0, (0, 255, 0), 2)

        cv.imshow("src", src)
        cv.imshow('edge', edge)
        cv.imshow("dst", dst)


        cv.waitKey()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
