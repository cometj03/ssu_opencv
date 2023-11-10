import cv2 as cv
import sys

"""
체커보드 n by n 구하기
"""

def main():
    filename = "board1.jpg" if len(sys.argv) < 2 else sys.argv[1]
    src = cv.imread(filename, cv.IMREAD_GRAYSCALE)

    if src is None:
        print("Image load failed!")
        exit()

    width = 700
    ratio = width / src.shape[1]
    inter_flag = cv.INTER_CUBIC if (ratio > 1) else cv.INTER_AREA
    src = cv.resize(src, (0, 0), fx=ratio, fy=ratio, interpolation=inter_flag)

    blur = cv.GaussianBlur(src, (0, 0), 1.2)
    # _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 71, 5)

    # cv.imshow("thresh", thresh)

    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 3))
    dilate = cv.morphologyEx(thresh, cv.MORPH_DILATE, horizontal_kernel, iterations=1)
    dilate = cv.morphologyEx(dilate, cv.MORPH_CLOSE, horizontal_kernel, iterations=2)
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 5))
    dilate = cv.morphologyEx(dilate, cv.MORPH_DILATE, vertical_kernel, iterations=1)
    dilate = cv.morphologyEx(dilate, cv.MORPH_CLOSE, vertical_kernel, iterations=2)

    contoured = cv.cvtColor(dilate, cv.COLOR_GRAY2BGR)
    # hierarchy : [Next, Previous, First_Child, Parent]
    cnts, hierarchy = cv.findContours(dilate, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    # 가장 많은 child의 개수 구하기
    top_level_idx = 0
    max_child_count = 0
    while top_level_idx >= 0:
        _child_idx = hierarchy[0, top_level_idx, 2]
        _max = 0
        while _child_idx >= 0:
            if cv.contourArea(cnts[_child_idx]) > 400:
                _max += 1
                cv.drawContours(contoured, [cnts[_child_idx]], 0, (0, 255, 0), 2)
            _child_idx = hierarchy[0, _child_idx, 0]
        max_child_count = max(max_child_count, _max)
        top_level_idx = hierarchy[0, top_level_idx, 0]

    cv.imshow("contours", contoured)

    N = round((max_child_count * 2)**0.5)
    print(f"{N}x{N}")

    cv.waitKey()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()