import pickle
from typing import Sequence, Dict, Tuple

import cv2 as cv
import numpy as np
from cv2.typing import MatLike, Point2f

FeatureList = list[Tuple[Tuple[int, int], Point2f, np.ndarray]]

train_img_paths = [
    "train_img/esb1.png",
    "train_img/esb2.png",
    "train_img/esb4.png",
    "train_img/esb5.png",
    "train_img/esb6.png",
    "train_img/esb7.png",
    "train_img/esb8.png",
]

wrong_img_paths = [
    "train_img/wrong/wrong1.png",
    "train_img/wrong/wrong2.jpg",
    "train_img/wrong/wrong3.jpg",
    "train_img/wrong/wrong4.jpg",
    "train_img/wrong/wrong5.png",
]


def waitKey() -> bool:
    c = cv.waitKey()
    cv.destroyAllWindows()
    if c == 27:
        return True
    return False


# 특징점 강화
# esb끼리의 공통점 추출
def get_empire_state_features(visualize: bool = False) -> FeatureList:
    visualize = visualize and not RENEW

    train_kp_list: list[Sequence[cv.KeyPoint]] = []
    train_desc_list: list[np.ndarray] = []

    src_list: list[MatLike] = []

    # detector: cv.Feature2D = cv.SIFT.create()
    # detector: cv.Feature2D = cv.KAZE.create()
    detector: cv.Feature2D = cv.ORB.create()
    matcher: cv.DescriptorMatcher = cv.BFMatcher.create(cv.NORM_HAMMING)
    # matcher: cv.DescriptorMatcher = cv.BFMatcher.create(cv.NORM_L2)

    # 모든 훈련 이미지의 특징점과 기술자 추출
    for path in train_img_paths:
        src = cv.imread(path, cv.IMREAD_GRAYSCALE)

        if src is None:
            print(f"Image load failed! path={path}")
            return

        kp, desc = detector.detectAndCompute(src, None)

        src_list.append(src)
        train_kp_list.append(kp)
        train_desc_list.append(desc)

    cnt = len(train_desc_list)

    features: FeatureList = list()

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
            # matches: Sequence[cv.DMatch] = matcher.match(desc1, desc2)
            # sorted_matches = sorted(matches, key=lambda x: x.distance)
            # good_matches = sorted_matches[:50]

            # homography 방식
            # kp1 = train_kp_list[i]
            # kp2 = train_kp_list[j]
            # pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
            # pts2 = np.array([kp2[m.trainIdx].pt for m in matches])
            #
            # H, mask = cv.findHomography(pts1, pts2, cv.RANSAC)
            # mask = mask.ravel()
            # vote = vote + mask
            #
            # good_matches = []
            # for k in range(mask.shape[0]):
            #     if mask[k]:
            #         good_matches.append(matches[k])

            # KNN 방식
            matches: Sequence[Sequence[cv.DMatch]] = matcher.knnMatch(desc1, desc2, k=2)
            good_matches: list[cv.DMatch] = []
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)

            for m in good_matches:
                vote[m.queryIdx] += 1

        ### end for j
        # good_pts: list[Point2f] = []  # keypoint의 위치만 저장. KeyPoint 객체를 pickling 할 수 없기 때문
        # good_desc: list[np.ndarray] = []

        # 시각화 용
        kps: Dict[int, list[cv.KeyPoint]] = dict()

        vote = vote.tolist()
        print('vote', vote)
        for v in range(desc1.shape[0]):
            if vote[v] == 0:
                continue
            pt: Point2f = train_kp_list[i][v].pt
            desc = train_desc_list[i][v]

            # if vote[v] not in features:  # key가 없으면 init
            #     features[vote[v]] = []
            # features[vote[v]].append((src_list[i].shape[:2], pt, desc))

            features.append((src_list[i].shape[:2], pt, desc))

            if visualize:
                if vote[v] not in kps:
                    kps[vote[v]] = []
                kps[vote[v]].append(train_kp_list[i][v])

        if visualize:
            for k in kps.keys():
                print('k', k)
                good_kp_dst = cv.drawKeypoints(src_list[i], kps[k], None, (-1, -1, -1),
                                               cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv.imshow("good kp dst", good_kp_dst)
                if waitKey():
                    break
    return features


def except_wrong_features(features: FeatureList, visualize: bool = False) -> FeatureList:
    visualize = visualize and not RENEW

    wrong_kp_list: list[Sequence[cv.KeyPoint]] = []
    wrong_desc_list: list[np.ndarray] = []

    wrong_src_list: list[MatLike] = []

    detector: cv.Feature2D = cv.ORB.create()
    matcher: cv.DescriptorMatcher = cv.BFMatcher.create(cv.NORM_HAMMING)

    for path in wrong_img_paths:
        src = cv.imread(path, cv.IMREAD_GRAYSCALE)

        if src is None:
            print(f"Image load failed! path={path}")
            return

        kp, desc = detector.detectAndCompute(src, None)

        wrong_src_list.append(src)
        wrong_kp_list.append(kp)
        wrong_desc_list.append(desc)

    shapes, pts, descs = zip(*features)
    descs = np.array(descs)

    exclude_mask = np.zeros((descs.shape[0],))

    for i in range(len(wrong_img_paths)):
        wrong_desc = wrong_desc_list[i]
        matches = matcher.knnMatch(wrong_desc, descs, k=2)

        good_matches: list[cv.DMatch] = []
        for m, n in matches:
            if m.distance < 0.85 * n.distance:
                good_matches.append(m)
                exclude_mask[m.trainIdx] += 1

        if visualize:
            kps = [wrong_kp_list[i][m.queryIdx] for m in good_matches]
            res = cv.drawKeypoints(wrong_src_list[i], kps, None, (-1, -1, -1),
                                   flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            print(len(good_matches))
            cv.imshow('wrong', res)
            waitKey()

    new_shapes, new_pts, new_descs = [], [], []
    for i in range(exclude_mask.shape[0]):
        if exclude_mask[i] < 1:
            new_shapes.append(shapes[i])
            new_pts.append(pts[i])
            new_descs.append(descs[i])

    return list(zip(new_shapes, new_pts, new_descs))


FILENAME = "esb_feature_vector.pkl"
VISUALIZE = False  # 시각화 여부
RENEW = True  # 파일 갱신 여부


# RENEW = False  # 파일 갱신 여부


def init():
    features = get_empire_state_features(visualize=VISUALIZE)
    print(len(features))
    features = except_wrong_features(features, visualize=True)
    print(len(features))

    if RENEW:
        with open(FILENAME, "wb") as f:
            pickle.dump(features, f)
        print('갱신 완료')


if __name__ == '__main__':
    init()
