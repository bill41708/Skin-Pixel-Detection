import os
import numpy as np
from PIL import Image
from sklearn.naive_bayes import GaussianNB
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')


Skin = []
Labelskin = []

path = './train/'

#取出為skin和不為skin的train data,並分別label為 1和0,存入list

for filename in os.listdir(path):
    if(os.path.isfile(path + filename) and not filename.startswith('.')):
        flag = 0
        #print(filename)
        if filename[0] == 'n':
            flag = 0
        else:
            flag = 1
        img = Image.open(path + filename)
        w, h = img.size
        for i in range(w):
            for j in range(h):
                [r, g, b] = img.getpixel((i,j))
                Skin.append([int(r), int(g), int(b)])
                Labelskin.append(flag)

Train = np.array(Skin)
Label = np.array(Labelskin)

print(Train.shape)
#print(Label.shape)

model = GaussianNB()
model.fit(Train, Label)

#predict所有的照片並分別印出結果

path1 = './Faces/'
for filename in os.listdir(path1):
    if(os.path.isfile(path1 + filename) and not filename.startswith('.')):


        img1 = Image.open(path1 + filename)
        w, h = img1.size
        print(w, h)
        count = 0
        count1 = 0
        test = []
        index = []
        for i in range(w):
            for j in range(h):
                [r, g, b] = img1.getpixel((i, j))
                test.append([int(r), int(g), int(b)])
                index.append([i, j])
                
                #predicted1 = model.predict(test)
                '''
                if(predicted1 == 1):
                    count += 1
                else:
                    count1 += 1
                '''

        predicted1 = model.predict(test)
        #print(predicted1.shape)



        plt.figure(figsize=(h, w), dpi=(1))
        plt.axis("off")

        index = np.array(index)
        #print(index.shape)
        #print(len(index))



        for i in range(len(index)):
            flag = 0
            if predicted1[i] == 1:
                flag = 254
            img1.putpixel((index[i][0], index[i][1]), (flag, flag, flag))
        img1.save('./result/' + filename, "JPEG")


'''
img = Image.open('skin1.jpg')
print(img.mode)
w,h = img.size
print(w,h)

arr = np.array(img)
print(arr.shape)
print(arr.dtype)
#print(arr)
'''


