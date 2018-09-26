from path_set import *
import sys
sys.path.append(PROJECT_PATH) # 添加项目位置至默认位置
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

annFile='{}\\annotations\\stuff_{}.json'.format(DATA_PATH,VAL_DATA_TYPE)

if __name__ == '__main__':
    # 初始化coco模型
    coco=COCO(annFile)

    # 显示COCO数据的类别信息
    cats = coco.loadCats(coco.getCatIds())
    print(cats)

    # get all images containing given categories, select one at random
    catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
    imgIds = coco.getImgIds(catIds=catIds );
    imgIds = coco.getImgIds(imgIds = [324158])

    # loadImgs导入图片
    img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

    print(img)
