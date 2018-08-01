# -*- coding:utf-8 -*-

from unet_camvid import *
from data_camvid import *

myunet = myUnet()
model = myunet.get_unet()
model.load_weights('unet_camvid.hdf5')

# test2mask
imgs_train, imgs_mask_train, imgs_test = myunet.load_data()
imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
np.save('./results/camvid_mask_test.npy', imgs_mask_test)

# mask2pic
myunet.save_img()
