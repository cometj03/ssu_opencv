import pickle
from typing import Sequence

import cv2 as cv
import numpy as np

from extract_esb_feature_vectors import FeatureList


def waitKey() -> bool:
    c = cv.waitKey()
    cv.destroyAllWindows()
    if c == 27:
        return True
    return False


def _resize(src):
    width, height = 1500, 800
    width_ratio = width / src.shape[1]
    height_ratio = height / src.shape[0]

    if width_ratio < 1 or height_ratio < 1:
        ratio = min(width_ratio, height_ratio)
        src = cv.resize(src, (0, 0), fx=ratio, fy=ratio, interpolation=cv.INTER_AREA)
    return src


FILENAME = "esb_feature_vector.pkl"

test_paths = [
    "img/esb1.jpg",
    "img/esb2.jpg",
    "img/esb3.jpg",
    "img/esb4.jpg",
    "img/esb5.jpg",
    "img/esb6.jpg",
    "img/esb7.jpg",
]


def main():
    # TODO 파일 없을 때 오류처리

    features: FeatureList
    # features: FeatureDict = get_empire_state_features()

    with open(FILENAME, "rb") as f:
        features = pickle.load(f)

    # p = test_paths[0]
    # test(p, features)

    i = 1
    for p in test_paths:
        print(i)
        i += 1
        test(p, features)
        if waitKey():
            break


def test(path: str, features: FeatureList):
    # filename = "img/esb2.jpg" if len(sys.argv) < 2 else sys.argv[1]

    _, _, esb_desc = zip(*features)
    esb_desc = np.array(esb_desc)

    dst = cv.imread(path, cv.IMREAD_GRAYSCALE)

    if dst is None:
        print(f'Image load failed! path={path}')
        return

    dst = _resize(dst)

    detector: cv.Feature2D = cv.ORB.create()
    matcher: cv.DescriptorMatcher = cv.BFMatcher.create(cv.NORM_HAMMING)

    dst_kp, dst_desc = detector.detectAndCompute(dst, None)

    good_matches: list[cv.DMatch] = []
    good_kps: list[cv.KeyPoint] = []

    # matches: Sequence[Sequence[cv.DMatch]] = matcher.knnMatch(esb_desc, dst_desc, k=2)
    #
    # for m, n in matches:
    #     if m.distance < 0.9 * n.distance:
    #         good_matches.append(m)
    #         good_kps.append(dst_kp[m.trainIdx])

    matches: Sequence[cv.DMatch] = matcher.match(esb_desc, dst_desc)
    sorted_matches = sorted(matches, key=lambda x: x.distance)
    good_matches = sorted_matches[:100]

    for m in good_matches:
        good_kps.append(dst_kp[m.trainIdx])

    print(len(matches))
    print(len(good_matches))

    res = cv.drawKeypoints(dst, good_kps, None, (-1, -1, -1), flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv.imshow("res", res)
    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
