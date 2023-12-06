import cv2 as cv
import numpy as np


def _resize(src):
    width, height = 1500, 800
    width_ratio = width / src.shape[1]
    height_ratio = height / src.shape[0]

    if width_ratio < 1 or height_ratio < 1:
        ratio = min(width_ratio, height_ratio)
        src = cv.resize(src, (0, 0), fx=ratio, fy=ratio, interpolation=cv.INTER_AREA)
    return src


# desc_t에서 desc_q와 같거나 비슷한 부분을 1로 하는 레이블을 반환하는 함수입니다
def get_label(
        desc_q: np.ndarray,
        desc_t: np.ndarray,
        matcher: cv.DescriptorMatcher
) -> np.ndarray:
    matches = matcher.knnMatch(desc_q, desc_t, k=2)

    label = np.zeros((len(desc_t),), dtype=np.int32)  # svm에 들어갈 label. 1이면 esb의 특징점이고, 0이면 배경 또는 다른 빌딩

    # m이 적절한 매칭이라면 m과 n의 각각의 distance 값 차이가 크다고 생각
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            label[m.trainIdx] = 1

    return label


def accuracy(predict_desc: np.ndarray, answer_path: str, detector: cv.Feature2D, matcher: cv.DescriptorMatcher):
    answer_src = cv.imread(answer_path, cv.IMREAD_GRAYSCALE)
    kp_a, desc_a = detector.detectAndCompute(answer_src, None)

    test_src = cv.imread("svm/5.png", cv.IMREAD_GRAYSCALE)
    kp_test, desc_test = detector.detectAndCompute(test_src, None)

    matches = matcher.knnMatch(desc_a, predict_desc, k=2)

    good_matches = []
    correct = 0
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            correct += 1
            good_matches.append(m)

    res = cv.drawMatches(answer_src, kp_a, test_src, kp_test, good_matches, None, (-1, -1, -1))
    res = _resize(res)
    cv.imshow("res", res)
    cv.waitKey()
    cv.destroyAllWindows()

    print(correct)
    print(predict_desc.shape[0])
    return correct / predict_desc.shape[0]


# 전체 이미지에서 추출한 특징점 중 empire state 빌딩의 특징점과 아닌 특징점을 분류하여 SVM으로 학습 시킵니다.
def main():
    path = [
        # ("svm/1_part.png", "svm/1.png"),
        # ("svm/2_part.png", "svm/2.png"),
        # ("svm/3_part.png", "svm/3.png"),
        ("svm/4_part.png", "svm/4.png"),
    ]

    detector: cv.Feature2D = cv.SIFT.create()
    matcher: cv.DescriptorMatcher = cv.BFMatcher.create(cv.NORM_L2)

    # detector: cv.Feature2D = cv.ORB.create()
    # matcher: cv.DescriptorMatcher = cv.BFMatcher.create(cv.NORM_HAMMING)

    # SVM 학습
    svm = cv.ml.SVM.create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setKernel(cv.ml.SVM_RBF)
    svm.setC(2.5)
    svm.setGamma(1e-05)

    q_path, t_path = path[0]
    # for q_path, t_path in path:
    print(q_path)
    src_query = cv.imread(q_path, cv.IMREAD_GRAYSCALE)
    src_train = cv.imread(t_path, cv.IMREAD_GRAYSCALE)

    kp_q, desc_q = detector.detectAndCompute(src_query, None)
    kp_t, desc_t = detector.detectAndCompute(src_train, None)

    for _ in range(2):
        label = get_label(desc_q, desc_t, matcher)

        desc_t = desc_t.astype(np.float32)
        svm.train(desc_t, cv.ml.ROW_SAMPLE, label)

        # svm.trainAuto(desc_t, cv.ml.ROW_SAMPLE, train_label)
        # c, gamma = svm.getC(), svm.getGamma()
        # print(c, gamma)

        # src_test = src_train.copy()
        src_test = cv.imread("svm/5.png", cv.IMREAD_GRAYSCALE)
        kp_test, desc_test = detector.detectAndCompute(src_test, None)

        pred_esb_kp, pred_not_esb_kp = [], []
        pred_desc = []
        for i in range(desc_test.shape[0]):
            test = np.array([desc_test[i]], dtype=np.float32)
            _, res = svm.predict(test)
            if res == 0:
                pred_not_esb_kp.append(kp_test[i])
            elif res == 1:
                pred_esb_kp.append(kp_test[i])
                pred_desc.append(desc_test[i])

        acc = accuracy(np.array(pred_desc), "svm/5_part.png", detector, matcher)
        print(pred_desc)
        print(acc)

        desc_q = np.array(pred_desc)

        res = cv.drawKeypoints(src_test, pred_esb_kp, None, (0, 0, 255))
        res = cv.drawKeypoints(res, pred_not_esb_kp, None, (0, 255, 255))
        res = _resize(res)
        cv.imshow(t_path, res)

    # cv.waitKey()
    # cv.destroyAllWindows()


if __name__ == '__main__':
    main()
