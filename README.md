### 使用COCO-QA数据集

初始化导入COCO数据的VQA的API

- vqa=VQA(annFile, quesFile)

获取问题ID

- annIds = vqa.getQuesIds(quesTypes='how many')

获取问题的所有可能回答

- anns = vqa.loadQA(annIds)