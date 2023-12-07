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

    shapes, kp, desc = zip(*features)
    return shapes, kp, np.array(desc)


def main():
    esb_dir = "img/empire_state/"
    others_dir = "img/others/"
    esb_path = {
        esb_dir + str(i) + ".jpg" for i in range(1, 11)
    }
    not_esb_path = {
        others_dir + str(i) + ".jpg" for i in range(1, 11)
    }
    # esb_path = {esb_dir + f for f in os.listdir(esb_dir)}
    # not_esb_path = {others_dir + f for f in os.listdir(others_dir)}
    test_path = list(esb_path.union(not_esb_path))

    shapes, esb_pts, esb_desc = get_esb_features()

    detector: cv.Feature2D = cv.SIFT.create()
    matcher: cv.DescriptorMatcher = cv.BFMatcher.create(cv.NORM_L2)

    pred_esb, pred_not_esb = 0, 0

    for path in test_path:
        src_test = cv.imread(path, cv.IMREAD_GRAYSCALE)

        if src_test is None:
            print(f"Image load failed! path={path}")
            return

        kp, desc = detector.detectAndCompute(src_test, None)

        # matches = matcher.match(esb_desc, desc)
        # good_matches = sorted(matches, key=lambda x: x.distance)[:100]

        matches = matcher.knnMatch(esb_desc, desc, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)

        if path in esb_path:
            pred_esb += len(good_matches)
        if path in not_esb_path:
            pred_not_esb += len(good_matches)

        (h, w) = shapes[0]
        feature_pts = [esb_pts[m.queryIdx] for m in good_matches]
        pts = [kp[m.trainIdx].pt for m in good_matches]

        H, _ = cv.findHomography(np.array(feature_pts), np.array(pts), cv.RANSAC)

        if H is None:
            print("continue")
            continue

        corners1 = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2).astype(np.float32)
        corners2 = cv.perspectiveTransform(corners1, H)

        res = cv.cvtColor(src_test, cv.COLOR_GRAY2BGR)
        cv.polylines(res, [np.int32(corners2)], True, (0, 0, 255), 3, cv.LINE_AA)

        #####
        match_kp = [kp[m.trainIdx] for m in good_matches]
        res = cv.drawKeypoints(res, match_kp, None, (0, 255, 0),
                               flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        res = _resize(res)
        cv.imshow('res', res)
        if cv.waitKey() == 27:
            break
        #####
    print("esb", pred_esb / len(esb_path))
    print("not esb", pred_not_esb / len(not_esb_path))


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()
