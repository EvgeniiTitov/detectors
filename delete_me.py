import cv2
import numpy as np
import random


def main():
    N = 15
    for i in range(N):
        c1 = np.ones((1080, 1920), dtype=np.uint8) * random.randint(0, 255)
        c2 = np.ones((1080, 1920), dtype=np.uint8) * random.randint(0, 255)
        c3 = np.ones((1080, 1920), dtype=np.uint8) * random.randint(0, 255)
        arr = cv2.merge((c1, c2, c3))
        cv2.imshow("", arr)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
