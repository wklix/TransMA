from predict import MolPredict
import pandas as pd

# 初始化模型并进行预测
clf = MolPredict(load_model='./raw_cliff', visual=False)
test_pred = clf.predict('./dataset/RaW/cliff/test.csv')

# 读取原始数据以获取 'TARGET' 列
#data = pd.read_csv('./dataset/waibushuju/waibu.csv')

# 创建包含预测结果的 DataFrame，并加入 'smiles' 和 'TARGET' 列
test_results = pd.DataFrame({
    'pred': test_pred.flatten(),
    'smiles': clf.datahub.data['smiles']
    #'label': data['TARGET']
})

# 打印前几行查看结果
print(test_results.head())

# 保存结果到新的 CSV 文件
test_results.to_csv("./del.csv", index=False)

