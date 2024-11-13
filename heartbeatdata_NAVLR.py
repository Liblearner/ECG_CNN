import numpy as np
import pandas as pd
import wfdb

"""
按心拍分割 筛选出MIT-BIH数据集中注释为[‘N’, ‘A’, ‘V’, ‘L’, ‘R’]的数据作为本次数据集，然后按照9：1的比例划分为训练集，验证集。最后送入卷积神经网络模型进行训练。
对心电图的标注样式如上图，“A"代表心房早搏，”."代表正常。整个数据集标注有40多种符号，表示40多种心拍状态。
df['label'].value_counts()
"""

# 五分类还是八分类
# ecgClassSet = ['N', 'A', 'V', 'L', 'R']
ecgClassSet = ['N', 'A', 'V', 'L', 'R', 'J', 'S', 'E']
numberSet = ['100', '101', '103', '105', '106', '108', '109', '111', '112', '113', '114', '115', '116', '117', '118',
             '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '207', '208', '209', '210', '212',
             '213', '214', '215', '219', '220', '221', '222', '223', '228', '230', '231', '232', '233', '234']
dataset, labelset = [], []

for number in numberSet:
    record = wfdb.rdrecord('./mit-bih-arrhythmia-database-1.0.0/' + number, channel_names=['MLII'])
    sig = record.p_signal.flatten()
    annotation = wfdb.rdann('./mit-bih-arrhythmia-database-1.0.0/' + number, 'atr')
    R_location = annotation.sample
    R_class = annotation.symbol

    start = 20
    end = 20
    i = start
    j = len(annotation.symbol) - end
    while i < j:
        try:
            label = ecgClassSet.index(R_class[i])
            data = sig[R_location[i] - 100:R_location[i] + 200]
            dataset.append(data)
            labelset.append(label)
            i += 1
        except ValueError:
            i += 1


dataset = np.array(dataset).reshape(-1, 300)
labelset = np.array(labelset).reshape(-1,1)
#将标签集 labelset 转换成NumPy数组，并将其形状重塑为(-1, 1)，其中-1同样表示根据数据推断这个维度的大小，而1表示每个样本只有一个标签。
print("8个分类的标签总个数: {}".format(np.size(labelset)))

print(f"N波的数量：{np.count_nonzero(labelset == 0.0)}")
print(f"A波的数量：{np.count_nonzero(labelset == 1.0)}")
print(f"V波的数量：{np.count_nonzero(labelset == 2.0)}")
print(f"L波的数量：{np.count_nonzero(labelset == 3.0)}")
print(f"R波的数量：{np.count_nonzero(labelset == 4.0)}")
print(f"J波的数量：{np.count_nonzero(labelset == 5.0)}")
print(f"S波的数量：{np.count_nonzero(labelset == 6.0)}")
print(f"E波的数量：{np.count_nonzero(labelset == 7.0)}")
print(np.shape(dataset))
print(np.shape(labelset))
dlset = np.hstack((dataset, labelset))
#: 使用np.hstack()函数将数据集和标签集水平堆叠在一起，形成一个包含特征数据和标签的数组。
# 转表格形式
columns = ['signals_' + str(x + 1) for x in range(301)]
#创建了一个包含301个字符串的列表，用于构建数据框架的列名。列名从 'signals_1' 到 'signals_301'。
df = pd.DataFrame(data=dlset, columns=columns)
#使用 Pandas 库的 DataFrame 函数，将dlset数组转换为一个数据框架，并指定列名为之前创建的columns列表。
df.rename(columns={"signals_301": 'label'}, inplace=True)
#在数据框架中将名为 'signals_301' 的列重命名为 'label'，这通常表示数据的类别或标签。inplace=True 表示在原始数据框架上进行修改，而不是创建一个副本。
print(df)
# 写入CSV文件（分开还是写一个表格待定）
df.to_csv(r"./NAVLR_rawdata_310.csv", header=True, index=False, sep=",")

