import tensorflow.keras as keras

#input data size맞게 처리하도록 convnet구성
from tensorflow.keras import layers

from tensorflow.keras import models

model=models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1))) #28by28 size 1channel(color있으면 3) input_shape를 first layer에 전달, 3by3 filter 32개, padding(x), strides=(1,1) defult
#model.add(layers.Conv2D(32,(3,3), activation='relu', padding="same", input_shape=(28,28,1))) #input과 ouput shape을 맞추거나 원하는 size 조절, padding="same"은 zero padding
model.add(layers.MaxPooling2D(2,2)) #2by2 pooling
model.add(layers.Conv2D(64,(3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64,(3, 3), activation='relu'))

#summary model architecture
model.summary()  #padding(x) 

###############convolutional layer########end################


model.add(layers.Flatten()) #3Dfilter > vector
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax")) #output10(mnist), softmax 분류값 내보내서 cross entropy계산

model.summary() #flatten: 마지막 layer output(3,3,64) > vector > 3*3*64=576

#################model#############end####################


#data download

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32')/255

train_labels = to_categorical(train_labels) #onehot
test_labels = to_categorical(test_labels)

#model complie

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10, batch_size=128)

#test data 검증

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(test_acc)

#fully connected network > 97.8%
#convolutional network > over 99%
