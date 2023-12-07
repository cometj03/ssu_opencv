import os
import sys

import cv2 as cv
import numpy as np

template_dir = "template/"
template_paths = [template_dir + f for f in os.listdir(template_dir)]


# test_dir = "img/test/"
# test_paths = [test_dir + f for f in os.listdir(test_dir)]


def predict(template, src2, visualize):
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
    # corners2 = corners2 + np.float32([w, 0])

    if visualize:
        res = template.copy()
        # res = cv.drawMatches(res, kp1, src2, kp2, good_matches, None, (-1, -1, -1),
        #                      flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        cv.polylines(res, [np.int32(corners2)], True, (0, 0, 255), 5, cv.LINE_AA)
        res = cv.resize(res, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
        cv.imshow("res", res)
        cv.waitKey()

    if cv.contourArea(corners2) < 1000:
        return False
    return True


def main():
    # test_path = "img/others/1.jpg" if len(sys.argv) < 2 else sys.argv[1]
    test_path = "img/test/9.jpg" if len(sys.argv) < 2 else sys.argv[1]
    src2 = cv.imread(test_path, cv.IMREAD_GRAYSCALE)

    if src2 is None:
        print(f"Image load failed! path={test_path}")
        return

    true_cnt = 0

    for path in template_paths:
        template = cv.imread(path, cv.IMREAD_GRAYSCALE)
        # src2 = cv.imread("img/others/3.jpg", cv.IMREAD_GRAYSCALE)
        if template is None:
            print(f"Image load failed! path={path}")
            return

        # template2 = cv.GaussianBlur(template, (0, 0), sigmaX=1.0)
        template2 = cv.bilateralFilter(template, -1, 10, 5)
        # src = cv.GaussianBlur(src2, (0, 0), sigmaX=1.0)
        src = cv.bilateralFilter(src2, -1, 10, 5)
        b = predict(template2, src, False)
        if b:
            true_cnt += 1

    if true_cnt >= len(template_paths) // 2:
        print(True)
    else:
        print(False)


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()
