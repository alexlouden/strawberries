import numpy as np
import matplotlib.pyplot as plt
import cv2

# read images
img1 = cv2.imread('1.png')
img2 = cv2.imread('2.png')

#convert read images to rgb for plotting
image = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)



#plot both test and result images
plt.figure(1)
plt.suptitle('Strawberry Detector', fontsize=16)

plt.subplot(821)
plt.title('1.png')
plt.imshow(image)

plt.subplot(825)
plt.imshow(img2)
plt.title('2.png')


plt.show()