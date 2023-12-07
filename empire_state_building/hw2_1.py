import os
import sys

import cv2 as cv
import numpy as np

template_dir = "template/"
template_paths = [template_dir + f for f in os.listdir(template_dir)]


def detect(template, src2):
    detector = cv.SIFT.create()
    matcher = cv.BFMatcher.create()

    kp1, desc1 = detector.detectAndCompute(template, None)
    kp2, desc2 = detector.detectAndCompute(src2, None)

    matches = matcher.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 10:
        return False

    pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)
    pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)

    H, _ = cv.findHomography(pts1, pts2, cv.RANSAC, 5.0)

    if H is None:
        return False

    (h, w) = template.shape[:2]
    corners1 = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2).astype(np.float32)
    corners2 = cv.perspectiveTransform(corners1, H)

    if cv.contourArea(corners2) < 2000:
        return False

    return True


def main():
    if len(sys.argv) < 2:
        print("파일 경로를 입력해주세요!")
        return
    test_path = sys.argv[1]
    src2 = cv.imread(test_path, cv.IMREAD_GRAYSCALE)

    if src2 is None:
        print(f"Image load failed! path={test_path}")
        return

    true_cnt = 0

    for path in template_paths:
        template = cv.imread(path, cv.IMREAD_GRAYSCALE)
        if template is None:
            print(f"Image load failed! path={path}")
            return

        template2 = cv.bilateralFilter(template, -1, 10, 5)
        src = cv.bilateralFilter(src2, -1, 10, 5)
        b = detect(template2, src)
        if b:
            true_cnt += 1

    if true_cnt >= 3:
        print(True)
    else:
        print(False)


if __name__ == '__main__':
    main()
    # cv.destroyAllWindows()
