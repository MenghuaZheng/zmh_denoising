# Matplotlib局部放大图画法
# 画Set12的图，需要图片的形状为正方形

import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


x1 = 30
x2 = 60

image_name = 'gt_kodim01.png'

# 数据准备
# image = plt.imread('./image/noise_kodim01.png')    #输入图片路径，如果是在当前工作路径下可以只写xx.jpg
image = Image.open(os.path.join('../input_img/', image_name))
np_image = np.array(image)

img_w, img_h, _ = np_image.shape
width, hight = int(img_w/10), int(img_w/10)

np_image_part = np_image[x2:x2+hight, x1:x1+width, :]
img_part = Image.fromarray(np_image_part)

# 绘制主图
# fig, ax = plt.subplots(1, 1)

# fig = plt.figure(figsize=(10, 10))
fig = plt.figure(figsize=(10, 10))

ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
ax.imshow(image)


ax.add_patch(plt.Rectangle((x1, x2), width, hight, color="red", fill=False, linewidth=3))

# 添加局部放大图
axins = ax.inset_axes((0.005, 0.005, 0.395, 0.395))
axins.imshow(img_part)

ax.add_patch(plt.Rectangle((int(0.7*img_w), int(0.7*img_h)), int(0.3*img_w), int(0.3*img_w), color="blue", fill=False, linewidth=10))

plt.axis('off')
axins.axis('off')

# plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
# plt.margins(0, 0)

plt.savefig(os.path.join('../output_img/', 'x1_{}_x2_{}_'.format(x1, x2) + image_name))

plt.show()


# 在缩放图中也绘制主图所有内容，然后根据限制横纵坐标来达成局部显示的目的