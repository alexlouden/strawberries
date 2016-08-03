from __future__ import division
import cv2
import numpy as np

green = (0, 255, 0)


def overlay_mask(mask, image):
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    return img


def find_biggest_contour(image):

    # Copy
    image = image.copy()
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Isolate largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]

    if not contour_sizes:
        return None, image

    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask


def circle_contour(image, contour):

    if contour is None:
        return image

    # Bounding ellipse
    image_with_ellipse = image.copy()
    ellipse = cv2.fitEllipse(contour)
    cv2.ellipse(image_with_ellipse, ellipse, green, 2, cv2.CV_AA)
    return image_with_ellipse


def process(image):

    image = cv2.resize(image, None, fx=1/3, fy=1/3)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Blur
    image_blur = cv2.GaussianBlur(image, (7, 7), 0)
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

    # Filter by colour
    # 0-10 hue
    min_red = np.array([0, 100, 80])
    max_red = np.array([10, 256, 256])
    mask1 = cv2.inRange(image_blur_hsv, min_red, max_red)

    # 170-180 hue
    min_red2 = np.array([170, 100, 80])
    max_red2 = np.array([180, 256, 256])
    mask2 = cv2.inRange(image_blur_hsv, min_red2, max_red2)

    # Combine masks
    mask = mask1 + mask2

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    # Find biggest strawberry
    big_strawberry_contour, mask_strawberries = find_biggest_contour(mask_clean)

    # Overlay cleaned mask on image
    overlay = overlay_mask(mask_clean, image)

    # Circle biggest strawberry
    circled = circle_contour(overlay, big_strawberry_contour)

    # Finally convert back to BGR to display
    bgr = cv2.cvtColor(circled, cv2.COLOR_RGB2BGR)

    return bgr


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
