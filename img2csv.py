import numpy as np
import pandas as pd
import os
import rasterio
import tqdm
from sklearn.model_selection import train_test_split

# ================================================================================================
def read_image(img_path):
    img_data = rasterio.open(img_path).read()
    band, height, width = np.shape(img_data)

    img_data_list = []
    for x in tqdm.trange(height):
        for y in range(width):
            temp = img_data[::, x, y]
            if np.array(np.isnan(temp), dtype=np.int8).sum() > 0:
                continue
            else:
                img_data_list.append(temp.tolist())

    img_arr = np.array(img_data_list)
    img_arr = np.around(img_arr, 6)
    labels = img_arr[:, 0]
    dataset = img_arr[:, 1:]
    print(os.path.basename(img_path), '读取成功!')

    # return dataset, labels
    return img_arr

# 主函数
train_data_path = './IMGTrain'

arr2d = [[0] * 22 for i in range(1)]
total_dataset = np.array(arr2d)

for img_name in os.listdir(train_data_path):  # [130036_20150111.tif, ...]
    path = train_data_path + '/' + img_name
    data = read_image(path)
    total_dataset = np.append(total_dataset, data, axis=0)
total_dataset = np.delete(total_dataset, 0, axis=0)
print(total_dataset, '\n', np.shape(total_dataset))
print('------------------------------------')

# 一张影像22个波段，每一波段为一种特征，特征名如下。 FSC即是模型训练时的标签数据也是模型输出数据
feature_name = ['FSC', 'SR1', 'SR2', 'SR3', 'SR4', 'SR5', 'SR6', 'SR7', 'NDVI', 'NDSI', 'NDFSI',
                'SensorZenith', 'SensorAzimuth', 'SolarZenith', 'SolarAzimuth',
                'Dem', 'Slope', 'Aspect', 'LST', 'A2T', 'SC', 'LCT']
df = pd.DataFrame(total_dataset, columns=feature_name)
# df.to_csv('./Data/NPP_Total.csv', index=False)
# df.to_csv('./Data/total_data.csv', index=False)
print(df)

train_data, test_data = train_test_split(df, test_size=0.3, random_state=1)
train_data.to_csv('./Data/train_data.csv', index=False)
test_data.to_csv('./Data/test_data.csv', index=False)
print(train_data)
print(test_data)

# """每张影像导出为csv"""
# data_path = '../IMGDataset/DataIMG/VNP_IMG_NF'
#
# for fn in os.listdir(data_path):
#     d_n = data_path + '/' + fn
#     out_n = './Data/Each/' + fn.split('.')[0] + '.csv'
#
#     data_1 = read_image(d_n)
#     feature_name = ['FSC', 'SR1', 'SR2', 'SR3', 'SR4', 'SR5', 'SR6', 'SR7', 'NDVI', 'NDSI', 'NDFSI',
#                     'SensorZenith', 'SensorAzimuth', 'SolarZenith', 'SolarAzimuth',
#                     'Dem', 'Slope', 'Aspect', 'LST', 'A2T', 'SC', 'LCT']
#     tdf = pd.DataFrame(data_1, columns=feature_name)
#
#     tdf.to_csv(out_n, float_format='%.6f', index=False)
#     print('----------------', fn, 'has done ----------------')
