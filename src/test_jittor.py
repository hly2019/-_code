import jittor as jt
import pylab as pl
import os
import numpy as np
img_path="./tmp/cat.jpg"

img = pl.imread(img_path)

# pl.subplot(121)
# pl.imshow(img)
kernel = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1],
])
# pl.subplot(122)
print(kernel.shape)
x = img[np.newaxis,:,:,:1].astype("float32")
print(type(x))
w = kernel[:,:,np.newaxis,np.newaxis].astype("float32")
# y = conv_naive(x, w)
# print (x.shape, y.shape) # shape exists confusion
# pl.imshow(y[0,:,:,0])
def conv_naive(x, w):
    N,H,W,C = x.shape

    Kh, Kw, _C, Kc = w.shape
    assert C==_C, (x.shape, w.shape)
    y = np.zeros([N,H-Kh+1,W-Kw+1,Kc])
    for i0 in range(N):
        for i1 in range(H-Kh+1): 
            for i2 in range(W-Kw+1):
                for i3 in range(Kh):
                    for i4 in range(Kw):
                        for i5 in range(C):
                            for i6 in range(Kc):
                                if i1-i3<0 or i2-i4<0 or i1-i3>=H or i2-i4>=W: continue
                                y[i0, i1, i2, i6] += x[i0, i1 + i3, i2 + i4, i5] * w[i3,i4,i5,i6]
    return y

def conv(x, w):
    N,H,W,C = x.shape
    Kh, Kw, _C, Kc = w.shape
    assert C==_C
    xx = x.reindex([N,H-Kh+1,W-Kw+1,Kh,Kw,C,Kc], [
        'i0', # Nid
        'i1+i3', # Hid+Khid
        'i2+i4', # Wid+KWid
        'i5', # Cid|
    ])
    ww = w.broadcast_var(xx)
    yy = (xx-ww) ** 2
    print(yy.shape)
    y = yy.sum([3,4,5]) # Kh, Kw, c
    # print("y is: {}".format(y.argmax(dim=0)))
    return y

# Let's disable tuner. This will cause jittor not to use mkl for convolution
jt.flags.enable_tuner = 0

# jx = jt.array(x)
# jw = jt.array(w)
# jy = conv(jx, jw).fetch_sync()
x = img[np.newaxis,:,:,:1].astype("float32")
w = kernel[:,:,np.newaxis,np.newaxis].astype("float32")
y = conv_naive(x, w)
# print (jx.shape, jy.shape)
# help(jt.numpy)