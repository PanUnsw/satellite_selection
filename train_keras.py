import keras
from keras.models import Sequential
from keras.layers import Conv2D,Dense, Dropout, Activation,MaxPooling2D,Flatten
from keras.layers import Reshape
from keras.layers import LSTM
from keras import backend as  backend
from reader import Star_Reader
from sklearn.preprocessing import OneHotEncoder

max_num_instars = Star_Reader.max_num_instars # 30
input_channel = Star_Reader.input_channel
input_shape = [max_num_instars,input_channel,1]
num_classes = 1

def model_LSTM():
    model = Sequential()
    model.add(LSTM(128))
    model.add(Dropout(0.9))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy',
                optimizer='Adam',
                metrics=['accuracy'])
    return model

def model_CNN():
    model = Sequential()
    #input: [b,n,c,1]]
    model.add( Conv2D(32,kernel_size=(1,input_channel),
                      activation='relu',
                      input_shape=input_shape) )
    # [b,n,1,32]
    model.add( Conv2D(64,(1,1),activation='relu') )
    model.add( Conv2D(num_classes,(1,1),activation='relu') )
    model.add( Reshape((-1)) )
    #model.add( Reshape((-1,num_classes)) )
    #model.add( Conv2D(num_classes,(1,1),activation='relu') )
    #backend.squeeze(model,axis=2)
 #   model.add( MaxPooling2D(pool_size=(2,1)) )
 #   model.add( Dropout(0.25) )
 #   model.add( Flatten() )
 #   model.add( Dense(128,activation='relu'))
 #   model.add( Dense(max_num_instars,activation='softmax') )
#    model.compile(optimizer='rmsprop',
#                 loss='categorical_crossentropy',
#                 metrics=['accuracy'])
    #model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model

model = model_CNN()

star_reader = Star_Reader()

x = star_reader.get_input().astype('float32')
x = backend.expand_dims(x, axis=-1)
y = star_reader.get_gt().astype('int32')

#enc = OneHotEncoder()
#enc.fit(y)
y_onehot = backend.one_hot(y,2)

x_train = x
y_train = y
test_num = 16*3
x_test = x[-test_num-1:-1,...]
y_test = y_onehot[-test_num-1:-1,...]
#model.fit(x_train, y_train, batch_size=16, epochs=10,verbose=1,validation_split=0.3)
model.fit(x_train, y_train, batch_size=16, epochs=10,verbose=1,validation_data=(x_train,y_train))
#score = model.evaluate(x_test, y_test, batch_size=16)
