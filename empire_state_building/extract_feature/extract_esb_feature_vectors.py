import os
import pickle
from typing import Sequence, Tuple

import cv2 as cv
import numpy as np
from cv2.typing import MatLike, Point2f

FeatureList = Sequence[Tuple[Tuple[int, int], Point2f, np.ndarray]]

train_img_paths = ["correct/" + f for f in os.listdir("correct")]
print(train_img_paths)


def waitKey() -> bool:
    c = cv.waitKey()
    if c == 27:
        return True
    return False


def _resize(src):
    width, height = 170, 600
    width_ratio = width / src.shape[1]
    height_ratio = height / src.shape[0]

    src = cv.resize(src, (width, height), interpolation=cv.INTER_CUBIC)
    # if width_ratio < 1 or height_ratio < 1:
    #     src = cv.resize(src, (0, 0), fx=height_ratio, fy=width_ratio, interpolation=cv.INTER_AREA)
    return src


# 특징점 강화
# esb끼리의 공통점 추출
def get_empire_state_features(visualize: bool = False) -> FeatureList:
    visualize = visualize and not RENEW

    train_kp_list: list[Sequence[cv.KeyPoint]] = []
    train_desc_list: list[np.ndarray] = []

    src_list: list[MatLike] = []

    detector: cv.Feature2D = cv.SIFT.create()
    # detector: cv.Feature2D = cv.KAZE.create()
    matcher: cv.DescriptorMatcher = cv.BFMatcher.create(cv.NORM_L2)

    # detector: cv.Feature2D = cv.ORB.create()
    # matcher: cv.DescriptorMatcher = cv.BFMatcher.create(cv.NORM_HAMMING)

    # 모든 훈련 이미지의 특징점과 기술자 추출
    for path in train_img_paths:
        src = cv.imread(path, cv.IMREAD_GRAYSCALE)

        if src is None:
            print(f"Image load failed! path={path}")
            return

        src = _resize(src)
        kp, desc = detector.detectAndCompute(src, None)

        src_list.append(src)
        train_kp_list.append(kp)
        train_desc_list.append(desc)

    cnt = len(train_img_paths)

    features: FeatureList = list()

    #
    for i in range(cnt):
        print('first', i)

        desc1 = train_desc_list[i]

        print(desc1.shape)
        vote = np.zeros((desc1.shape[0],), dtype=np.int32)

        for j in range(cnt):
            if i == j:
                continue
            print('second', j)

            desc2 = train_desc_list[j]

            # 브루트포스 방식
            matches: Sequence[cv.DMatch] = matcher.match(desc1, desc2)
            sorted_matches = sorted(matches, key=lambda x: x.distance)
            good_matches = sorted_matches[:100]

            # KNN 방식
            # matches: Sequence[Sequence[cv.DMatch]] = matcher.knnMatch(desc1, desc2, k=2)
            # good_matches: list[cv.DMatch] = []
            # for m, n in matches:
            #     if m.distance < 0.7 * n.distance:
            #         good_matches.append(m)

            #####
            if visualize:
                res = cv.drawMatches(src_list[i], train_kp_list[i], src_list[j], train_kp_list[j],
                                     good_matches, None, (-1, -1, -1), flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv.imshow('matches', res)
                if waitKey():
                    break
            #####

            for m in good_matches:
                vote[m.queryIdx] += 1

        ### end for j

        kps: list[cv.KeyPoint] = list()  # 시각화용

        vote = vote.tolist()
        print('vote', vote)
        for v in range(desc1.shape[0]):
            if vote[v] <= cnt // 2 - 1:  # 과반수 이상 공통될 때
                continue
            pt: Point2f = train_kp_list[i][v].pt
            desc = train_desc_list[i][v]

            features.append((src_list[i].shape[:2], pt, desc))

            if visualize:
                kps.append(train_kp_list[i][v])

        if visualize:
            good_kp_dst = cv.drawKeypoints(src_list[i], kps, None, (-1, -1, -1),
                                           cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv.imshow("good kp dst", good_kp_dst)
            if waitKey():
                break
    return features


FILENAME = "esb.pkl"
VISUALIZE = True  # 시각화 여부
RENEW = True  # 파일 갱신 여부


def init():
    if RENEW:
        input()
    features = get_empire_state_features(visualize=VISUALIZE)
    print(len(features))

    if RENEW:
        with open(FILENAME, "wb") as f:
            pickle.dump(features, f)
        print('갱신 완료')


if __name__ == '__main__':
    init()
    cv.destroyAllWindows()
