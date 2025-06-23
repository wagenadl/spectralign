# testplacement.py - part of spectralign

import spectralign
import numpy as np
import matplotlib.pyplot as plt

img1 = spectralign.Image("C2-20250523_montage_YFP_ideal_02.jpg")
img2 = spectralign.Image("C2-20250523_montage_YFP_ideal_03.jpg")

p1, p2, swm = spectralign.refine(img1, img2, (1000, 600), (200, 600), 400, 3)

plt.figure(1)
plt.clf()
f, ax = plt.subplots(1, 2, num=1)
ax[0].imshow(img1)
ax[0].plot(p1[0], p1[1], 'w+')
ax[1].imshow(img2)
ax[1].plot(p2[0], p2[1], 'w+')

plc = spectralign.Placement()
plc.addmatch(0, 1, p1, p2)
pos = plc.solve()

rnd = spectralign.Render(pos, img1.shape)
rnd.render(0, img1, apo=300)
rnd.render(1, img2, apo=300)
img = rnd.image

plt.figure(2)
plt.clf()
plt.imshow(img)

rndx = spectralign.Render(pos, img1.shape)
rndx.render(0, img1, apo=300, hard=False)
rndx.render(1, img2, apo=300, hard=False)
imgx = rndx.blended()

plt.figure(3)
plt.clf()
plt.imshow(imgx)
