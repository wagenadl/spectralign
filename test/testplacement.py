# testplacement.py - part of spectralign

import spectralign
import numpy as np
import matplotlib.pyplot as plt

img1 = spectralign.Image("C2-20250523_montage_YFP_ideal_02.jpg")
img2 = spectralign.Image("C2-20250523_montage_YFP_ideal_03.jpg")
 
def alignpair(sta0, xsta, ysta,
              mov0, xmov, ymov,
              R):
    sta = sta0.straightwindow([xsta, ysta], R)
    mov = mov0.straightwindow([xmov, ymov], R)
    swm = spectralign.Swim().align(sta, mov)
    dx, dy = swm.shift
    xsta -= dx/2
    ysta -= dy/2
    xmov += dx/2
    ymov += dy/2
    sta = sta0.straightwindow([xsta, ysta], R)
    mov = mov0.straightwindow([xmov, ymov], R)
    swm1 = spectralign.Swim().align(sta, mov)
    dx, dy = swm1.shift
    xsta -= dx/2
    ysta -= dy/2
    xmov += dx/2
    ymov += dy/2
    sta = sta0.straightwindow([xsta, ysta], R)
    mov = mov0.straightwindow([xmov, ymov], R)
    swm2 = spectralign.Swim().align(sta, mov)
    if False:
        print(ix, iy, fmt(swm.shift), fmt(swm1.shift), fmt(swm2.shift),
              fmt(swm.snr), fmt(swm1.snr), fmt(swm2.snr),
              fmt(swm.shift + swm1.shift + swm2.shift))
    dx, dy = swm2.shift
    xsta -= dx/2
    ysta -= dy/2
    xmov += dx/2
    ymov += dy/2
    return xsta, ysta, xmov, ymov, swm2.snr

xsta, ysta, xmov, ymov, snr = alignpair(img1, 1000, 600,
                                        img2, 200, 600,
                                        400)

plt.figure(1)
plt.clf()
f, ax = plt.subplots(1, 2, num=1)
ax[0].imshow(img1)
ax[0].plot(xsta, ysta, 'w+')
ax[1].imshow(img2)
ax[1].plot(xmov, ymov, 'w+')

plc = spectralign.Placement()
plc.addmatch(0, 1, (xsta, ysta), (xmov, ymov))
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
