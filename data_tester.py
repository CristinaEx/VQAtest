from vqaTools.vqa import VQA
import random
import skimage.io as io
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('D:\\VQA\\PythonHelperTools')
from path_set import *

annFile='{}\\annotations\\{}{}_{}_annotations.json'.format(DATA_PATH,VERSION_TYPE,DATA_TYPE,VAL_DATA_TYPE)
quesFile ='{}\\Questions\\{}{}_{}_{}_questions.json'.format(DATA_PATH,VERSION_TYPE,TASK_TYPE,DATA_TYPE,VAL_DATA_TYPE)

if __name__ == '__main__':
    # initialize VQA api for QA annotations
    vqa=VQA(annFile, quesFile)
    # load and display QA annotations for given question types
    """
    All possible quesTypes for abstract and mscoco has been provided in respective text files in ../QuestionTypes/ folder.
    """
    annIds = vqa.getQuesIds(quesTypes='how many');   
    anns = vqa.loadQA(annIds)
    randomAnn = random.choice(anns)
    vqa.showQA([randomAnn])
    imgId = randomAnn['image_id']
    imgFilename = 'COCO_' + VAL_DATA_TYPE + '_'+ str(imgId).zfill(12) + '.jpg'
    if os.path.isfile(VAL_DATA_PATH + imgFilename):
	    I = io.imread(VAL_DATA_PATH + imgFilename)
	    plt.imshow(I)
	    plt.axis('off')
	    plt.show()

    # load and display QA annotations for given answer types
    """
    ansTypes can be one of the following
    yes/no
    number
    other
    """
    annIds = vqa.getQuesIds(ansTypes='yes/no');   
    anns = vqa.loadQA(annIds)
    randomAnn = random.choice(anns)
    vqa.showQA([randomAnn])
    imgId = randomAnn['image_id']
    imgFilename = 'COCO_' + VAL_DATA_TYPE + '_'+ str(imgId).zfill(12) + '.jpg'
    if os.path.isfile(VAL_DATA_PATH + imgFilename):
	    I = io.imread(VAL_DATA_PATH + imgFilename)
	    plt.imshow(I)
	    plt.axis('off')
	    plt.show()

    # load and display QA annotations for given images
    """
    Usage: vqa.getImgIds(quesIds=[], quesTypes=[], ansTypes=[])
    Above method can be used to retrieve imageIds for given question Ids or given question types or given answer types.
    """
    ids = vqa.getImgIds()
    annIds = vqa.getQuesIds(imgIds=random.sample(ids,5));  
    anns = vqa.loadQA(annIds)
    randomAnn = random.choice(anns)
    vqa.showQA([randomAnn])  
    imgId = randomAnn['image_id']
    imgFilename = 'COCO_' + VAL_DATA_TYPE + '_'+ str(imgId).zfill(12) + '.jpg'
    if os.path.isfile(VAL_DATA_PATH + imgFilename):
	    I = io.imread(VAL_DATA_PATH + imgFilename)
	    plt.imshow(I)
	    plt.axis('off')
	    plt.show()
    