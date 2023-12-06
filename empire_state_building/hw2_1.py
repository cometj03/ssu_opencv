import pickle
from typing import Sequence, Tuple

import cv2 as cv
import numpy as np
from cv2.typing import Point2f


def _resize(src):
    width, height = 1500, 1000
    width_ratio = width / src.shape[1]
    height_ratio = height / src.shape[0]

    if width_ratio < 1 or height_ratio < 1:
        ratio = min(width_ratio, height_ratio)
        src = cv.resize(src, (0, 0), fx=ratio, fy=ratio, interpolation=cv.INTER_AREA)
    return src


def get_esb_features(filename: str = "extract_feature/esb.pkl"):
    FeatureList = Sequence[Tuple[Tuple[int, int], Point2f, np.ndarray]]
    features: FeatureList

    with open(filename, "rb") as f:
        features = pickle.load(f)

    _, _, desc = zip(*features)
    return np.array(desc)


def main():
    test_path = [
        "img/esb1.jpg",
        "img/esb2.jpg",
        "img/esb3.jpg",
        "img/esb4.jpg",
        "img/esb5.jpg",
        "img/esb6.jpg",
        "img/esb7.jpg",
    ]
    not_esb_path = [
        "img/not-esb1.jpg",
        "img/not-esb2.jpg",
        "img/not-esb3.jpg",
        "img/not-esb4.jpg",
        "img/not-esb5.jpg",
        "img/not-esb6.png",
        "img/not-esb7.jpg",
    ]
    test_path += not_esb_path

    detector: cv.Feature2D = cv.SIFT.create()
    matcher: cv.DescriptorMatcher = cv.BFMatcher.create(cv.NORM_L2)

    for path in test_path:
        src_test = cv.imread(path, cv.IMREAD_GRAYSCALE)

        kp, desc = detector.detectAndCompute(src_test, None)
        esb_desc = get_esb_features()

        # matches = matcher.match(esb_desc, desc)
        # good_matches = sorted(matches, key=lambda x: x.distance)[:100]

        matches = matcher.knnMatch(esb_desc, desc, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)

        match_kp = [kp[m.trainIdx] for m in good_matches]
        res = cv.drawKeypoints(src_test, match_kp, None, (0, 0, 255),
                               flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        res = _resize(res)
        cv.imshow('res', res)
        cv.waitKey()


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()
