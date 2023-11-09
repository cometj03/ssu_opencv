import cv2 as cv

def main():
    src = cv.imread("checker2.jpg", cv.IMREAD_GRAYSCALE)
    blur = cv.GaussianBlur(src, (13, 13), 0)
    _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    cv.imshow("thresh", thresh)

    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 5))
    dilate = cv.dilate(thresh, horizontal_kernel, iterations=2)
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 9))
    dilate = cv.dilate(dilate, vertical_kernel, iterations=2)

    # dilate = cv.dilate(thresh, None)
    # dilate = cv.dilate(thresh, None)
    cv.imshow("dilate2", dilate)

    print(horizontal_kernel)
    print(vertical_kernel)

    cv.waitKey()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()