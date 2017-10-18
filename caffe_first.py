import caffe

print(dir(caffe))
caffe.set_mode_cpu()

mynet = caffe.Net("first.prototxt", caffe.TEST)
print(mynet.inputs)
print(mynet.blobs)

import numpy as np

inp= np.zeros(1,1,100,100)
random= np.random.rand(100,100)

inp[0,0]=inp[0,0]+random
net.blobs['data'].data = inp
net.forward()
print(net.blobs['data'])

