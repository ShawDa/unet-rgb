# -*- coding:utf-8 -*-

from unet import *
from data import *

myunet = myUnet()
model = myunet.get_unet()
model.load_weights('unet.hdf5')

# test2mask
imgs_train, imgs_mask_train, imgs_test = myunet.load_data()
imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
np.save('./results/imgs_mask_test.npy', imgs_mask_test)

# mask2pic
myunet.save_img()
