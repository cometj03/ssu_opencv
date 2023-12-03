# 코드 임시 저장소
from typing import Union, Sequence

import cv2 as cv
import numpy as np
from cv2.typing import MatLike

not_esb_img_paths = [
    "img/not-esb-seoul.jpg",
    "img/not-esb-one_world_trade_center1.jpg",
    # "img/not-esb-one_world_trade_center3.jpg",
    # "img/not-esb-one_world_trade_center2.jpg",
]


def waitKey() -> bool:
    c = cv.waitKey()
    cv.destroyAllWindows()
    if c == 27:
        return True
    return False


def _getDescriptor(filename) -> Union[np.ndarray, None]:
    src: np.ndarray = cv.imread(filename, cv.IMREAD_GRAYSCALE)

    if src is None:
        print(f"Image load failed! (filename: {filename})")
        return None

    # detector = cv.SIFT_create()
    detector: cv.Feature2D = cv.KAZE_create()

    esb_kp: Sequence[cv.KeyPoint]
    esb_desc: np.ndarray
    esb_kp, esb_desc = detector.detectAndCompute(src, None)

    desc_count: int = esb_desc.shape[0]
    exclude_mask: np.ndarray = np.full((desc_count,), False)

    for p in not_esb_img_paths:
        not_esb_src: MatLike = cv.imread(p, cv.IMREAD_GRAYSCALE)

        not_esb_kp: Sequence[cv.KeyPoint]
        not_esb_desc: np.ndarray
        not_esb_kp, not_esb_desc = detector.detectAndCompute(not_esb_src, None)

        matcher: cv.DescriptorMatcher = cv.BFMatcher_create(cv.NORM_L2)
        matches: Sequence[cv.DMatch] = matcher.match(esb_desc, not_esb_desc)
        # matches = matcher.knnMatch(esb_desc, not_esb_desc, k=2)

        # good = []
        # for m, n in matches:
        #     if m.distance < 0.9 * n.distance:
        #         good.append(m)
        #         include_mask[m.queryIdx] = False

        max_dist = max(matches, key=lambda x: x.distance).distance
        min_dist = min(matches, key=lambda x: x.distance).distance
        mean_dist = (max_dist + min_dist) / 2

        print("max", max_dist)
        print("min", min_dist)
        print("mean", mean_dist)

        filtered_match = filter(lambda x: x.distance < mean_dist, matches)

        for m in filtered_match:
            exclude_mask[m.queryIdx] = True
        print(len(matches))
        print(np.count_nonzero(exclude_mask))

        # 시각화 용
        # sorted_match: list[cv.DMatch] = sorted(matches, key=lambda x: x.distance)
        # batch = 5
        # for i in range(0, len(sorted_match) - batch, batch):
        #     dst = cv.drawMatches(src, esb_kp, not_esb_src, not_esb_kp, sorted_match[i:i+batch], None)
        #     cv.imshow('match_dst', dst)
        #     if waitKey():
        #         break

    desc = esb_desc[exclude_mask]

    print('total', np.count_nonzero(exclude_mask))

    # 시각화 용
    kp = []
    for i in range(len(esb_kp)):
        if not exclude_mask[i]:
            kp.append(esb_kp[i])

    dst = cv.drawKeypoints(src, kp, None, (-1, -1, -1))
    dst_origin = cv.drawKeypoints(src, esb_kp, None, (-1, -1, -1))
    cv.imshow('dst', dst)
    cv.imshow('origin', dst_origin)

    return desc
