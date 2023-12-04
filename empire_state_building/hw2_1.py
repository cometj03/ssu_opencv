import pickle
from typing import Dict, Tuple, Sequence

import cv2 as cv
import numpy as np
from cv2.typing import Point2f

from extract_esb_feature_vectors import reinforced_features

"""
- key: 훈련 이미지 경로
- value: tuple(keypoint들의 위치 리스트, 기술자 리스트)
"""
FeatureDict = Dict[str, Tuple[Sequence[Point2f], Sequence[np.ndarray]]]


def waitKey() -> bool:
    c = cv.waitKey()
    cv.destroyAllWindows()
    if c == 27:
        return True
    return False


def _resize(src):
    width, height = 1500, 1000
    width_ratio = width / src.shape[1]
    height_ratio = height / src.shape[0]

    if width_ratio < 1 or height_ratio < 1:
        ratio = min(width_ratio, height_ratio)
        src = cv.resize(src, (0, 0), fx=ratio, fy=ratio, interpolation=cv.INTER_AREA)
    return src


def _getMatchesUsingBF_KNN(desc1, desc2):
    matcher = cv.BFMatcher_create(cv.NORM_L2)
    matches = matcher.knnMatch(desc1, desc2, k=2)
    return matches


def _getMatchesUsingFlann_KNN(desc1, desc2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc1, desc2, k=2)
    return matches


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

    features: FeatureDict = reinforced_features()
    # with open(FILENAME, "rb") as f:
    #     features = pickle.load(f)

    i = 1
    for p in test_paths:
        print(i)
        i += 1
        test(p, features)
        if waitKey():
            break


def test(path: str, features: FeatureDict):
    # filename = "img/esb2.jpg" if len(sys.argv) < 2 else sys.argv[1]

    dst = cv.imread(path, cv.IMREAD_GRAYSCALE)

    if dst is None:
        print(f'Image load failed! path={path}')
        return

    dst = _resize(dst)

    # detector: cv.Feature2D = cv.SIFT_create()
    # detector: cv.Feature2D = cv.KAZE_create()
    detector: cv.Feature2D = cv.ORB_create()
    matcher: cv.DescriptorMatcher = cv.BFMatcher_create(cv.NORM_L2)

    for k in features.keys():
        train_img = cv.imread(k, cv.IMREAD_GRAYSCALE)
        train_points, train_desc = features[k]
        train_desc = np.array(train_desc)
        train_points = np.array(train_points).reshape((-1, 1, 2)).astype(np.float32)
        train_kp = [cv.KeyPoint(p[0, 0], p[0, 1], 0, 0, 0, 0, 0) for p in train_points]

        dst_kp, dst_desc = detector.detectAndCompute(dst, None)

        matches: Sequence[cv.DMatch] = matcher.match(train_desc, dst_desc)
        # cnt = len(matches)
        # sorted_matches = sorted(matches, key=lambda x: x.distance)
        # good_matches = sorted_matches[:cnt // 2]

        # matches: Sequence[Sequence[cv.DMatch]] = matcher.knnMatch(feat_desc, desc, k=2)
        # good_matches: list[cv.DMatch] = []
        # for m, n in matches:
        #     if m.distance < 0.9 * n.distance:
        #         good_matches.append(m)

        dst_points = np.array([dst_kp[m.trainIdx].pt for m in matches]).reshape((-1, 1, 2)).astype(np.float32)
        H, mask = cv.findHomography(train_points, dst_points, cv.RANSAC)

        matchesMask = mask.ravel().tolist()

        res = cv.drawMatches(train_img, train_kp, dst, dst_kp, matches, None,
                             matchesMask=matchesMask,
                             flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        cv.imshow("asdf", res)
        # dst = cv.drawKeypoints(src, good_keypoints, None, (-1, -1, -1), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv.imshow("dst", dst)
        if waitKey():
            break


if __name__ == '__main__':
    main()
