import skimage.io,skimage.draw,skimage.feature
import numpy as np
import matplotlib.pyplot as plt



img = skimage.io.imread('eye.jpg')

#just use grayscale, but you could make separate template for each r,g,b channel 
img = np.mean(img, axis=2)

(M,N) = img.shape
mm = M-20
nn = N-20
template = np.zeros([mm,nn])

## Create template ##

#darkest inner circle (pupil)
(rr,cc) = skimage.draw.circle(mm/2,nn/2,4.5, shape=template.shape)
template[rr,cc]=-2

#iris (circle surrounding pupil)
(rr,cc) = skimage.draw.circle(mm/2,nn/2,8, shape=template.shape)
template[rr,cc] = -1

#Optional - pupil reflective spot (if centered)
(rr,cc) = skimage.draw.circle(mm/2,nn/2,1.5, shape=template.shape)
template[rr,cc] = 1

plt.imshow(template)

normccf = skimage.feature.match_template(img, template,pad_input=True)

#center pixel
(i,j) = np.unravel_index( np.argmax(normccf), normccf.shape)

plt.imshow(img)
plt.plot(j,i,'r*')
