# noise visualization

import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

image_name1 = 'out_kodim15.png'
image_name2 = 'noise_kodim15.png'
# 数据准备
# image = plt.imread('./image/noise_kodim01.png')    #输入图片路径，如果是在当前工作路径下可以只写xx.jpg
image1 = Image.open(os.path.join('../input_img/', image_name1))
image2 = Image.open(os.path.join('../input_img/', image_name2))
np_image1 = np.uint8(image1)
np_image2 = np.uint8(image2)

noise = (np_image2-np_image1).clip(0, 255)

image_noise = Image.fromarray(noise)

image_noise.save('../output_img/image_noise'+image_name2)