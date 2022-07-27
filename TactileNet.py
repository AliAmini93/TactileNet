from tensorflow.keras import backend as K
K.set_image_data_format('channels_first')

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout, multiply
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2,l1,l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.layers import concatenate
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

def Incp_EEGNet(nb_classes, Chans = 32, Samples = 128, kernLength = 16, F1 = 64, D=4, dropoutRate = 0.5,dropoutType = 'Dropout'):

 
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    #EEGNet alike part
    input1       = Input(shape = (1, Chans, Samples))
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                    input_shape = (1, Chans, Samples),
                                    use_bias = False)(input1)
    block1       = BatchNormalization(axis = 1)(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                    depth_multiplier = D,
                                    depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization(axis = 1)(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 2))(block1)
    block1       = dropoutType(dropoutRate)(block1)

    ###############################################
    #first tower
    sub_block1   = Conv2D(64, (1, 1), padding = 'same',use_bias = False)(block1)
    sub_block1   = SeparableConv2D(128, (1, 128), padding = 'same',use_bias = False)(sub_block1)
    sub_block1   = AveragePooling2D((1, 2), padding = 'same')(sub_block1)
    #second tower
    sub_block2   = Conv2D(16, (1, 1), padding = 'same',use_bias = False)(block1)
    sub_block2   = SeparableConv2D(32, (1, 256), padding = 'same',use_bias = False)(sub_block2)
    sub_block2  = AveragePooling2D((1, 2), padding = 'same')(sub_block2)
    #third tower
    sub_block3   = Conv2D(64, (1, 1), padding = 'same', strides=(1,2), use_bias = False)(block1)
    #forth tower
    sub_block4   = AveragePooling2D((1, 2), padding = 'same')(block1)
    sub_block4   = Conv2D(32, (1, 1), padding = 'same',use_bias = False)(sub_block4)
    #concatenation
    concat       = concatenate([sub_block1, sub_block2, sub_block4, sub_block3],axis=1)
    
    #last tower
    block2       = BatchNormalization(axis = 1)(concat)
    block2       = Activation('elu')(block2)
    #SENEt block
    squeeze1     = GlobalAveragePooling2D()(block2)
    excitation1  = Dense(16, activation='relu')(squeeze1)
    excitation1  = Dense(256, activation='sigmoid')(excitation1)
    block2       = Permute(dims=(2,3,1))(block2)
    excitation1  = multiply([block2, excitation1])
    excitation1  = Permute(dims=(3,1,2))(excitation1)

    block2       = SeparableConv2D(256, (1, 64), padding = 'same',use_bias = False)(excitation1)
    block2       = BatchNormalization(axis = 1)(block2)
    block2       = Activation('elu')(block2)
    #SENEt block
    squeeze2     = GlobalAveragePooling2D()(block2)
    excitation2  = Dense(16, activation='relu')(squeeze2)
    excitation2  = Dense(256, activation='sigmoid')(excitation2)
    block2       = Permute(dims=(2,3,1))(block2)
    excitation2  = multiply([block2, excitation2])
    excitation2  = Permute(dims=(3,1,2))(excitation2)

    block2       = dropoutType(dropoutRate)(excitation2)

    GB           = GlobalAveragePooling2D()(block2)
    dense        = Dense(nb_classes, name = 'dense')(GB)
    softmax      = Activation('softmax', name = 'softmax')(dense)

    return Model(inputs=input1, outputs=softmax)

import numpy as np
X_train = np.load('X_train.npy')
Y_train = np.load('Y_train.npy')

X_test = np.load('X_test.npy')
Y_test = np.load('Y_test.npy')

X_val = np.load('X_val.npy')
Y_val = np.load('Y_val.npy')
# EEGNet-specific imports
import numpy as np
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
kernels, chans, samples = 1, 32, 1024
#############################################################################
test_labels = Y_test
############################# EEGNet portion ##################################

# convert labels to one-hot encodings.
Y_train      = np_utils.to_categorical(Y_train - 1)
Y_val        = np_utils.to_categorical(Y_val - 1)
Y_test       = np_utils.to_categorical(Y_test - 1)

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import math
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

keras_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001, verbose=2)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

model = Incp_EEGNet(nb_classes=6, Chans = 32, Samples = 1024, kernLength = 512,
                    F1 = 64, D=4, dropoutRate = 0.1,
                    dropoutType = 'Dropout')
model.summary()
# compile the model and set the optimizers
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics = ['accuracy'])
# count number of parameters in the model
numParams    = model.count_params()    
print(numParams)
###############################################################################
# if the classification task was imbalanced (significantly more trials in one
# class versus the others) you can assign a weight to each class during 
# optimization to balance it out. This data is approximately balanced so we 
# don't need to do this, but is shown here for illustration/completeness. 
###############################################################################

# the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
# the weights all to be 1
fittedModel = model.fit(X_train, Y_train, batch_size = 32, epochs = 200, 
                        verbose = 2, validation_data=(X_val, Y_val), callbacks=[es, keras_reduce_lr])

probs       = model.predict(X_test)
preds       = probs.argmax(axis = -1)  
acc         = np.mean(preds == Y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))
import matplotlib.pyplot as plt
from pyriemann.utils.viz import plot_confusion_matrix
# plot the confusion matrices for both classifiers
names        = ['1_dynamic_left','1_dynamic_right', '2_dynamic_left','2_dynamic_right','3_dynamic_left','3_dynamic_right']
plt.figure(0)
plot_confusion_matrix(preds, Y_test.argmax(axis = -1), names, title = 'Tactile_Net')

'''
plt.plot(fittedModel.history['accuracy'])
plt.plot(fittedModel.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(fittedModel.history['loss'])
plt.plot(fittedModel.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
'''

# ROC curve
import scikitplot as skplt
import matplotlib.pyplot as plt
skplt.metrics.plot_roc_curve(test_labels, probs)
plt.show()
#########################################################
#F1 score
from sklearn.metrics import f1_score
Fscore_micro =f1_score(Y_test.argmax(axis = -1), preds, average='micro')
print("F1-Score micro: %f " % (Fscore_micro))
Fscore_macro =f1_score(Y_test.argmax(axis = -1), preds, average='macro')
print("F1-Score macro: %f " % (Fscore_macro))
########################################################
# kappa
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(Y_test.argmax(axis = -1), preds)
print('Cohens kappa: %f' % kappa)
