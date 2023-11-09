import cv2 as cv

"""
체커보드 n by n 구하기
"""

def main():
    src = cv.imread("checker4.jpg", cv.IMREAD_GRAYSCALE)

    # todo: resize 말고 perspective 트랜스폼 해서 정확도 높이기
    # 적응형 이진화 해야할지도

    # width = 700
    # ratio = width / src.shape[1]
    # inter_flag = cv.INTER_CUBIC if (ratio > 1) else cv.INTER_AREA
    # src = cv.resize(src, (0, 0), fx=ratio, fy=ratio, interpolation=inter_flag)

    blur = cv.GaussianBlur(src, (0, 0), 1.2)
    _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

    cv.imshow("thresh", thresh)

    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 3))
    # dilate = cv.dilate(thresh, horizontal_kernel, iterations=2)
    dilate = cv.morphologyEx(thresh, cv.MORPH_DILATE, horizontal_kernel, iterations=1)
    dilate = cv.morphologyEx(dilate, cv.MORPH_CLOSE, horizontal_kernel, iterations=2)
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 5))
    # dilate = cv.dilate(dilate, vertical_kernel, iterations=2)
    dilate = cv.morphologyEx(dilate, cv.MORPH_DILATE, vertical_kernel, iterations=1)
    dilate = cv.morphologyEx(dilate, cv.MORPH_CLOSE, vertical_kernel, iterations=2)

    cv.imshow("dilate", dilate)

    contoured = cv.cvtColor(dilate, cv.COLOR_GRAY2BGR)
    # hierarchy : [Next, Previous, First_Child, Parent]
    cnts, hierarchy = cv.findContours(dilate, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(contoured, cnts, 0, (0, 255, 0), 2, cv.LINE_8, hierarchy)

    # 가장 많은 child의 개수 구하기
    top_level_idx = 0
    max_child_count = 0
    while top_level_idx >= 0:
        _child_idx = hierarchy[0, top_level_idx, 2]
        _max = 0
        while _child_idx >= 0:
            _max += 1
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