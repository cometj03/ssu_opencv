import cv2 as cv
import sys

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

def main():
    # test("img/not-esb-chrysler1.jpg")
    # cv.waitKey()
    # cv.destroyAllWindows()

    for i in range(1, 8):
        test(f"img/esb{i}.jpg")

        c = cv.waitKey()
        cv.destroyAllWindows()
        if c == 27:
            break

def test(filename):
    # filename = "img/esb2.jpg" if len(sys.argv) < 2 else sys.argv[1]
    # src1 = cv.imread("esb1.png", cv.IMREAD_GRAYSCALE)  # queryImg
    src1 = cv.imread("train_img/esb3.png", cv.IMREAD_GRAYSCALE)  # queryImg
    src2 = cv.imread(filename, cv.IMREAD_GRAYSCALE)                     # trainImg

    if src1 is None or src2 is None:
        print('Image load failed!')
        return

    src2 = _resize(src2)

    # detector = cv.ORB_create()
    detector = cv.KAZE_create()
    # detector = cv.SIFT_create()
    keypoints1, desc1 = detector.detectAndCompute(src1, None)
    keypoints2, desc2 = detector.detectAndCompute(src2, None)
    print('desc1.shape:', desc1.shape)
    print('desc2.shape:', desc2.shape)
    print(desc1)


    kp_query = cv.drawKeypoints(src1, keypoints1, None, (-1, -1, -1),
                            cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kp_train = cv.drawKeypoints(src2, keypoints2, None, (-1, -1, -1),
                            cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv.imshow("keypoints_q", kp_query)
    cv.imshow("keypoints_t", kp_train)

    matches = _getMatchesUsingBF_KNN(desc1, desc2)
    # matches = _getMatchesUsingFlann_KNN(desc1, desc2)

    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)



    dst = cv.drawMatches(src1, keypoints1, src2, keypoints2, good, None)
    cv.imshow("match", dst)

    # cv.waitKey()
    # cv.destroyAllWindows()

if __name__ == '__main__':
    main()
