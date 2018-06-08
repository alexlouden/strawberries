# strawberry-detector 

Here demonstrated is a strawberry detector using color segementation and image filtering techniques. After detecting the approriate color it draws a contour around the most dense areas of the range that is being detected. 

# Dependencies 

The following dependencies were used : 
   * openCV
   * matplotlib
   * numpy
   * math
   
# Results

Below is the result for the processes that occurs in the back end of the script for detection. A mask is created and then it is being further processed to reduce noise. Finally morphiological transformation is done to get a more smooth mask which is followed by drawing a contour over the original input image which is shown at the output.<br /><br />
![alt-text-1](https://github.com/hasibzunair/strawberry-detector/blob/master/Figure_1.png "title-1")

Here are the results showing the input image and the contour drawn in the output image. There is no straighforward determination of how accurate this system is bacause this is just simply extracting a specific color range, red in this case.<br />

![alt-text-2](https://github.com/hasibzunair/strawberry-detector/blob/master/Figure_2.png "title-2")
![alt-text-2](https://github.com/hasibzunair/strawberry-detector/blob/master/Figure_3.png "title-2")
![alt-text-2](https://github.com/hasibzunair/strawberry-detector/blob/master/Figure_4.png "title-2")
![alt-text-2](https://github.com/hasibzunair/strawberry-detector/blob/master/Figure_5.png "title-2")
![alt-text-2](https://github.com/hasibzunair/strawberry-detector/blob/master/Figure_6.png "title-2")
![alt-text-2](https://github.com/hasibzunair/strawberry-detector/blob/master/Figure_7.png "title-2")
![alt-text-2](https://github.com/hasibzunair/strawberry-detector/blob/master/Figure_8.png "title-2")
![alt-text-2](https://github.com/hasibzunair/strawberry-detector/blob/master/Figure_9.png "title-2")


# Resources
Computer vision on strawberries
- Slides - [Keynote](Computer%20Vision%20Slides.key), [Powerpoint](Computer%20Vision%20Slides.pptx) or [PDF](Computer%20Vision%20Slides.pdf)
- Jupyter notebooks - [working](Strawberry%20working.ipynb) and [final with examples](Strawberry%20complete%20with%20examples.ipynb)
- Demos: [HSV video](demo_hsv_video.py) and [strawberry video](demo_strawberry_video.py)

# Final Output

![](strawberries_found.jpg)
