import cv2 as cv
import numpy as np


def waitKey() -> bool:
    c = cv.waitKey()
    cv.destroyAllWindows()
    if c == 27:
        return True
    return False


def main():
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    mask = arr > 3
    print(mask)


if __name__ == '__main__':
    main()
