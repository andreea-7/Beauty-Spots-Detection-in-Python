

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
from matplotlib import pyplot as plt1

fig=plt1.figure(figsize=(10,7))

rows=2
columns=1

#Enter path
image = cv2.imread("")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

fig.add_subplot(rows, columns,1)
plt1.imshow(image)
plt1.axis('off')
plt1.title("Poza originala")

image0 = cv2.split(image)

result_planes = []
result_norm_planes = []
for plane in image0:
    dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(plane, bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    result_planes.append(diff_img)
    result_norm_planes.append(norm_img)

result = cv2.merge(result_planes)
image0 = cv2.merge(result_norm_planes)

pixel_values = image0.reshape((-1, 3))
pixel_values = np.float32(pixel_values)
print(pixel_values.shape)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

k = 4
k=int(k)
p, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)

labels = labels.flatten()

segmented_image = centers[labels.flatten()]

segmented_image = segmented_image.reshape(image0.shape)

rgb_scale = segmented_image


height = rgb_scale.shape[0]
width = rgb_scale.shape[1]

r2=0
g2=0
b2=0
contor=0;

d=[0]*k

for i in centers:
    d[contor] = math.sqrt((r2 - i[0]) ** 2 + (g2 - i[1] ) ** 2 + (b2 - i[2]) ** 2)
    contor=contor+1

min1=d[0]
contor=0
k=contor
for i in d:
    if min1>i:
        min1=i
        k=contor
    contor=contor+1

print("culoarea este:", centers[k])

r0,g0,b0=centers[k]
p=0

a = [[0]*2]*(height*width)
contor=0
j=0
i=0
for i in range(height):
    for j in range(width):
      r1,g1,b1=rgb_scale[i][j]
      if r0==r1 and g0==g1 and b0==b1:
          a[contor]=[i,j]
      contor=contor+1

height_o=image.shape[0]
width_o=image.shape[1]

contor=0

for i in range(height_o):
    for j in range(width_o):
        [x,y]=a[contor]
        if x!=0 or y!=0:
            cv2.circle(image, (j,i), 1, (255, 0, 0), -1)
        contor=contor+1


fig.add_subplot(rows, columns,2)
plt1.imshow(image)
plt1.axis('off')
plt1.title("Poza cu nevii detectati")

plt1.show()