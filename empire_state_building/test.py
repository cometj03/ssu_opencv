from typing import Sequence

import cv2 as cv
import numpy as np


def waitKey() -> bool:
    c = cv.waitKey()
    cv.destroyAllWindows()
    if c == 27:
        return True
    return False


def main():
    # src = cv.imread("test2.png", cv.IMREAD_GRAYSCALE)
    src = cv.imread("img/not-esb-seoul.jpg", cv.IMREAD_GRAYSCALE)

    # detector: cv.Feature2D = cv.SIFT_create()
    # detector: cv.Feature2D = cv.KAZE_create()
    detector: cv.Feature2D = cv.ORB_create()

    kps: Sequence[cv.KeyPoint]
    kps, desc = detector.detectAndCompute(src, None)

    dst: np.ndarray = src.copy()

    kps = sorted(kps, key=lambda x: -x.response)
    for i in range(len(kps)):
        kp: cv.KeyPoint = kps[i]
        # print('point', kp.pt)
        # print('size', kp.size)
        # print('octave', kp.octave)
        # print('response', kp.response)
        # print(desc[i])

        # dst = cv.drawKeypoints(dst, [kp], None, (-1, -1, -1), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv.imshow('dst', dst)
        # if waitKey():
        #     break

    dst = cv.drawKeypoints(dst, kps, None, (-1, -1, -1), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow('dst', dst)
    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
