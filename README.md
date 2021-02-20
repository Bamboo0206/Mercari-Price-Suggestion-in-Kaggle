# Mercari-Price-Suggestion-in-Kaggle
该项目来自 kaggle 竞赛，根据商品的描述，品牌，品类，物品的状态等文本来预测商品的价格 。通过分析描述商品的包含文本和数值的表格，在数据清洗后使用正则表达式、Porter Stemmer、TF-IDF进行了特征工程，选择 Ridge、LigntGBM 和 MLP 三种模型进行训练，最后使用 MSLE 评估预测结果
