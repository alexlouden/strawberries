from __future__ import division
import cv2
import numpy as np

def process(img):

    image = cv2.resize(img, None, fx=1/3, fy=1/3)

    # RGB
    images = []
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i in [0, 1, 2]:
        colour = rgb.copy()
        if i != 0: colour[:,:,0] = 0
        if i != 1: colour[:,:,1] = 0
        if i != 2: colour[:,:,2] = 0
        images.append(colour)
    rgb_stack = np.vstack(images)
    rgb_stack = cv2.cvtColor(rgb_stack, cv2.COLOR_RGB2BGR)

    # HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    images = []
    for i in [0, 1, 2]:
        colour = hsv.copy()
        if i != 0: colour[:,:,0] = 0
        if i != 1: colour[:,:,1] = 255
        if i != 2: colour[:,:,2] = 255
        images.append(colour)

    hsv_stack = np.vstack(images)
    hsv_stack = cv2.cvtColor(hsv_stack, cv2.COLOR_HSV2BGR)

    both = np.hstack([rgb_stack, hsv_stack])

    return both


def main():

    # Load video
    video = cv2.VideoCapture(0)

    cv2.namedWindow("Video")

    if not video.isOpened():
        raise RuntimeError('Video not open')

    while True:
        f, img = video.read()

        result = process(img)

        cv2.imshow('Video', result)

        # Wait for 1ms
        key = cv2.waitKey(1) & 0xFF

        # Press escape to exit
        if key == 27:
            return

        # Reached end of video
        if not f:
            return

if __name__ == '__main__':
    main()
