from __future__ import division
import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import cos, sin

blue = (255, 0, 0)

def show(image):
    plt.figure(figsize=[10, 10])
    plt.imshow(image, interpolation='nearest')

def overlay_mask(mask, image):
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    return img

def find_biggest_contour(image):
    image = image.copy()

    _, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask

def circle_contour(image,contour):
    image_with_ellipse = image.copy()
    ellipse = cv2.fitEllipse(contour)
    cv2.ellipse(image_with_ellipse, ellipse, blue, 8, cv2.LINE_AA)

    return image_with_ellipse


def find_strawberry(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    max_dimension = max(image.shape)
    scale = 700/max_dimension
    image = cv2.resize(image, None, fx=scale, fy=scale)

    image_blur = cv2.GaussianBlur(image, (7,7), 0)
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

    min_red = np.array([0, 100, 80])
    max_red = np.array([10, 256, 256])

    mask1 = cv2.inRange(image_blur_hsv, min_red, max_red)

    min_red2 = np.array([170,100,80])
    max_red2 = np.array([180, 255, 256])

    mask2 = cv2.inRange(image_blur_hsv, min_red2, max_red2)

    mask = mask1+mask2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_CLOSE, kernel)

    big_strawberry_contour, mask_strawberries = find_biggest_contour(mask_clean)

    overlay = overlay_mask(mask_clean, image)
    circled = circle_contour(overlay, big_strawberry_contour)
    #show(circled)
    bgr = cv2.cvtColor(circled, cv2.COLOR_RGB2BGR)

    return bgr


in1 = cv2.imread('1.png')
in2 = cv2.imread('2.png')
in3 = cv2.imread('3.png')
in4 = cv2.imread('4.png')
in5 = cv2.imread('5.png')
in6 = cv2.imread('6.png')
in7 = cv2.imread('7.png')
in8 = cv2.imread('8.png')

in1_rgb = cv2.cvtColor(in1, cv2.COLOR_BGR2RGB)
in2_rgb = cv2.cvtColor(in2, cv2.COLOR_BGR2RGB)
in3_rgb = cv2.cvtColor(in3, cv2.COLOR_BGR2RGB)
in4_rgb = cv2.cvtColor(in4, cv2.COLOR_BGR2RGB)
in5_rgb = cv2.cvtColor(in5, cv2.COLOR_BGR2RGB)
in6_rgb = cv2.cvtColor(in6, cv2.COLOR_BGR2RGB)
in7_rgb = cv2.cvtColor(in7, cv2.COLOR_BGR2RGB)
in8_rgb = cv2.cvtColor(in8, cv2.COLOR_BGR2RGB)

result1 = find_strawberry(in1)
result2 = find_strawberry(in2)
result3 = find_strawberry(in3)
result4 = find_strawberry(in4)
result5 = find_strawberry(in5)
result6 = find_strawberry(in6)
result7 = find_strawberry(in7)
result8 = find_strawberry(in8)

cv2.imwrite('Detect1.png', result1)
cv2.imwrite('Detect2.png', result2)
cv2.imwrite('Detect3.png', result3)
cv2.imwrite('Detect4.png', result4)
cv2.imwrite('Detect5.png', result5)
cv2.imwrite('Detect6.png', result6)
cv2.imwrite('Detect7.png', result7)
cv2.imwrite('Detect8.png', result8)



plt.figure(1)
plt.suptitle('Strawberry Detector', fontsize=16)
plt.subplot(121)
plt.title('Test set/Input')
plt.imshow(in1_rgb)
plt.subplot(122)
plt.title('Result/Output')
plt.imshow(result1)

plt.figure(2)
plt.suptitle('Strawberry Detector', fontsize=16)
plt.subplot(121)
plt.title('Test set/Input')
plt.imshow(in2_rgb)
plt.subplot(122)
plt.title('Result/Output')
plt.imshow(result2)

plt.figure(3)
plt.suptitle('Strawberry Detector', fontsize=16)
plt.subplot(121)
plt.title('Test set/Input')
plt.imshow(in3_rgb)
plt.subplot(122)
plt.title('Result/Output')
plt.imshow(result3)

plt.figure(4)
plt.suptitle('Strawberry Detector', fontsize=16)
plt.subplot(121)
plt.title('Test set/Input')
plt.imshow(in4_rgb)
plt.subplot(122)
plt.title('Result/Output')
plt.imshow(result4)

plt.figure(5)
plt.suptitle('Strawberry Detector', fontsize=16)
plt.subplot(121)
plt.title('Test set/Input')
plt.imshow(in5_rgb)
plt.subplot(122)
plt.title('Result/Output')
plt.imshow(result5)

plt.figure(6)
plt.suptitle('Strawberry Detector', fontsize=16)
plt.subplot(121)
plt.title('Test set/Input')
plt.imshow(in6_rgb)
plt.subplot(122)
plt.title('Result/Output')
plt.imshow(result6)

plt.figure(7)
plt.suptitle('Strawberry Detector', fontsize=16)
plt.subplot(121)
plt.title('Test set/Input')
plt.imshow(in7_rgb)
plt.subplot(122)
plt.title('Result/Output')
plt.imshow(result7)

plt.figure(8)
plt.suptitle('Strawberry Detector', fontsize=16)
plt.subplot(121)
plt.title('Test set/Input')
plt.imshow(in8_rgb)
plt.subplot(122)
plt.title('Result/Output')
plt.imshow(result8)

plt.show()
