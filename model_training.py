# -*- coding:utf-8 -*-

import numpy as np
from PIL import Image, ImageFont, ImageDraw
import pandas as pd
import glob
import cv2  
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from pylab import *
from scipy.cluster.vq import *
from scipy.misc import imresize
from scipy import misc,ndimage
from skimage import measure,morphology,color

#包含的汉字列表（太长，仅仅截取了一部分） 5的3次方个
hanzi = u'等'

#生成文字矩阵
def gen_img(text, size=(48,48), fontname='simhei.ttf', fontsize=48,pos_x=0, pos_h=0, pos_size=0,pos_rangle=0):
#    im = Image.new("RGBA", size,(255, 255, 255))  #mode,size,color
    im = Image.new("RGB", size,1)  #mode,size,color  rgb为3*8像素
    dr = ImageDraw.Draw(im)
    font = ImageFont.truetype(fontname, fontsize)
    dr.text((0, 0), text, font=font)
    im2 = np.array(im)
   
    color = [  
            ([130, 130, 130],[255,255,255])#颜色范围~数值按[b,g,r]排布  
    ] 
    for (lower, upper) in color:  
        lower = np.array(lower, dtype = "uint8")#颜色下限  
        upper = np.array(upper, dtype = "uint8")#颜色上限  
        mask = cv2.inRange(im2, lower, upper) 
        mask_one = mask / 255
    h, w = mask.shape
    num_w_point = zeros([h])
    for i in range(h):
        num_w_point[i] = sum(mask_one[i,:])
    hight_start_index, hight_stop_index = cut_to_lines(num_w_point)
   
    num_h_point = zeros([w])
    for i in range(w):
        num_h_point[i] = sum(mask_one[:,i])     
    width_start_index, width_stop_index = cut_to_lines(num_h_point)
      
    hight_start=hight_start_index[0] #字的上边界
    hight_stop=hight_stop_index[len(hight_stop_index)-1]+1
    width_start=width_start_index[0]  #字的左边界
    width_stop=width_stop_index[len(width_stop_index)-1]+1
    
    if(pos_size == 0):   #处理左右
        image =gen_img_xy(mask,hight_start=hight_start,hight_stop=hight_stop,width_start=width_start,width_stop=width_stop,pos_x=pos_x, pos_h=pos_h, pos_size=pos_size)
    else:               #处理大小
        image =gen_img_size(mask,hight_start=hight_start,hight_stop=hight_stop,width_start=width_start,width_stop=width_stop,pos_x=pos_x, pos_h=pos_h, pos_size=pos_size)
   
    
    #开闭运算-定义结构元素 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2, 2))
    #开闭运算，先开运算去除背景噪声，再继续闭运算填充目标内的孔洞
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel) 
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel) 
    image = closed

    #图片旋转
    (h, w) = image.shape[:2]
    center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, pos_rangle, 1.0)
    image = cv2.warpAffine(image, M, (w, h)) 
    
    #加噪声
    coutn = 50
    for k in range(0,coutn):
        #get the random point
        xi = int(np.random.uniform(0,image.shape[1]))
        xj = int(np.random.uniform(0,image.shape[0]))
        #add noise
        image[xj,xi] = 255
        
    m4=Image.fromarray(image)
    return np.array(m4.getdata())
    
# k为每行点数，返回起始于结束的分割集合
def cut_to_lines(k):
    length = len(k)
    start_index = [i+1 for i in range(length-1)  if ((k[i+1] > 0) and ( (k[i]==0) or (i==0) ))]
    k2 = np.append(k,0)
    stop_index = [i for i in range(length) if ( (k2[i]>0) and (k2[i+1] == 0) ) ]
    return start_index, stop_index

#左右
def gen_img_xy(mask,hight_start=0,hight_stop=0,width_start=0,width_stop=0,pos_x=0, pos_h=0, pos_size=0):
    if(pos_x > 0):
        img_f2 = mask[:, width_start+pos_x :width_stop]  
    else:
        img_f2 = mask[:, width_start:width_stop + pos_x]
    if(pos_h > 0):   
        img_f2 = img_f2[hight_start+pos_h:hight_stop,:]
    else:
        img_f2 = img_f2[hight_start:hight_stop+pos_h,:]
    img_f3 = cv2.resize(img_f2,(48 , 48),cv2.INTER_LINEAR); #放大，像素自动填充
    #图像二值化 cv2.threshold  但是20是确定的。。
    ret,thresh1=cv2.threshold(img_f3,20,255,cv2.THRESH_BINARY)
    return thresh1
   

#size
def gen_img_size(mask,hight_start=0,hight_stop=0,width_start=0,width_stop=0,pos_x=0, pos_h=0, pos_size=0):
    img_f2 = mask[hight_start+pos_size : hight_stop-pos_size, width_start+pos_size : width_stop-pos_size]  
    img_f3 = cv2.resize(img_f2,(48 , 48),cv2.INTER_LINEAR); #放大，像素自动填充
    #图像二值化 cv2.threshold  但是20是确定的。。
    ret,thresh1=cv2.threshold(img_f3,20,255,cv2.THRESH_BINARY)
    return thresh1     


#生成训练样本
data = pd.DataFrame()
fonts = glob.glob('./*.[tT][tT]*')
for fontname in fonts:
    print(fontname)
    for pos_rangle in (-10,0,10): #图片旋转
        for pos_x in range (-2,2): #左右偏移
            for pos_h in range (-2,2):  #上下偏移 
                m = pd.DataFrame(pd.Series(list(hanzi)).apply(lambda s:[gen_img(s, fontname=fontname, fontsize=48, pos_x=pos_x, pos_h=pos_h,pos_size= 0,pos_rangle=pos_rangle)]))
                m['label'] = range(1)
                data = data.append(m, ignore_index=True) 
        for pos_size in range(-2,2):   #大小偏移
            m = pd.DataFrame(pd.Series(list(hanzi)).apply(lambda s:[gen_img(s, fontname=fontname, fontsize=48, pos_x=0, pos_h=0, pos_size=pos_size,pos_rangle=pos_rangle)]))
            m['label'] = range(1)
            data = data.append(m, ignore_index=True) 
        #append方法可以添加数据到一个dataframe中，注意append方法不会影响原来的dataframe，会返回一个新的dataframe。
        #ignore_index如果为True，会对新生成的dataframe使用新的索引（自动产生），忽略原来数据的索引。
#x = np.array(list(data[0])).astype(float)
y = np.array(list(data[0])).astype(float)
np.save('y', y) #保存训练数据

# =============================================================================
#查看生成的字
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import pandas as pd
import glob
img_rows, img_cols = 48, 48
x = np.load('y.npy')
x= x.reshape(x.shape[0], img_cols, img_rows,1)
def show_character(k):
    xx  = x[k,:,:]
    cv2.imshow("images", xx)  
    cv2.waitKey(0)
print(len(data)) 
#show_character(1)
# =============================================================================
for i in range(0, len(data) ):
    show_character(i);





"""
dic=dict(zip(range(100),list(hanzi))) #构建字表

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

batch_size = 1024
nb_classes = 100
nb_epoch = 30

img_rows, img_cols = 48, 48
# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 4

x = np.load('x.npy')
x= x.reshape(x.shape[0], img_cols, img_rows,1)
y = np_utils.to_categorical(list(range(100))*2*5*2, nb_classes)
weight = ((100-np.arange(100))/100.0+1)**3
weight = dict(zip(range(101),weight/weight.mean())) #调整权重，高频字优先

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(img_cols, img_rows, 1,)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x, y,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    class_weight=weight)

score = model.evaluate(x,y)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save_weights('model.model')
"""