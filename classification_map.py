
import numpy as np
import matplotlib.pyplot as plt

def plot(classifier, inp, out, ticks=200):
    # ranges
    xfrom = inp[:,0].min()*1.1
    xto = inp[:,0].max()*1.1
    yfrom = inp[:,1].min()*1.1
    yto = inp[:,1].max()*1.1

    # meshgrid
    h = (xto - xfrom) / ticks
    xx, yy = np.arange(xfrom, xto, h), np.arange(yfrom, yto, h)
    xx, yy = np.meshgrid(xx, yy)
    zz = np.empty(xx.shape, dtype=float)

    # classify meshgrid
    pos = 0
    for x in range(xx.shape[0]):
        for y in range(xx.shape[1]):
            zz[x][y] = classifier(xx[x][y], yy[x][y])

    plt.clf()
    plt.contourf(xx, yy, zz, alpha=0.5) # class separations
    plt.scatter(inp[:,0], inp[:,1], c=out, s=50) # dataset points
    plt.gca().set_aspect('equal', 'datalim')
    plt.show()