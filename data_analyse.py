import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyecharts.charts import Pie
from pyecharts import options as opts
from pyecharts.render import make_snapshot
from snapshot_phantomjs import snapshot
from minepy import MINE
import random
import rioxarray
import plotly.graph_objects as go
import plotly.io as pio

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 防止坐标轴上的负号乱码

# df = pd.read_csv('./Data/test_data.csv')
# print(df.head())
# print(df.describe())
# print(df['FSC'].values)

# plt.style.use('fivethirtyeight')
# sns.pairplot(df, hue='FSC', vars=df.columns[:], diag_kind='kde')
# plt.show()

# ---------- 数据集划分饼图 ----------
# c = Pie()
# c.add("", [['Train data', 37], ['Test data', 16]], radius=["30%", "75%"],)  # 设置圆环的粗细和大小
# c.set_global_opts(title_opts=opts.TitleOpts(title="MODIS 训练集划分"))
# c.set_series_opts(label_opts=opts.LabelOpts(formatter="{b}:{c}"))
# # c.render('Dataset Split.html')
# make_snapshot(snapshot, c.render(), "Dataset Split.png")

# plt.figure(dpi=150, figsize=(5, 4))
# plt.pie([7, 3], explode=[0.1, 0], labels=['Train data', 'Valid data'], colors=['lightcoral', 'lightskyblue'], autopct='%1.1f%%', shadow=True)
# plt.title('MODIS 训练集划分')
# plt.savefig('./images/Dataset_Split.svg', dpi=150, bbox_inches='tight', format='svg')
# plt.show()

# ---------- 特征的标准差（离散度）计算 ----------
# std = df.std()
# features = std.index.values
# val = std.values
# print(features)
# print(val)

# ---------- Pearson折线图 ----------
# pearson = df.corr()
# val = pearson.values
# print(val)

# pearson = df.corr()['FSC']
# feature = ['SR1', 'SR2', 'SR3', 'SR4', 'SR5', 'SR6', 'SR7', 'NDVI', 'NDSI', 'NDFSI',
#            'SensorZenith', 'SensorAzimuth', 'SolarZenith', 'SolarAzimuth',
#            'Dem', 'Slope', 'Aspect', 'LST', 'A2T', 'SC', 'LCT']
# val = pearson.values[1:]

# plt.figure(dpi=400, figsize=(10, 5))
# plt.plot(feature, val, 'bo-', alpha=1, linewidth=1, label='Pearson Correlation Coefficient')
# plt.legend()
# plt.xlabel('Feature')
# plt.ylabel('Value')
# plt.ylim(-1, 1)
# plt.xticks(rotation=60)
# plt.axhline(0, color='gray')
# plt.savefig('./images/Pearson Correlation Coefficient.svg', dpi=400, bbox_inches='tight', format='svg')
# plt.show()

# ---------- 热力图 ----------
# plt.figure(dpi=400, figsize=(12, 10))
# p1 = sns.heatmap(df.corr(method='pearson'), linewidths=0.1, vmax=1.0, square=True, linecolor='white', annot=True, annot_kws={'size': 7})
# plt.title('Pearson Heat Map')
# plt.show()
# s1 = p1.get_figure()
# s1.savefig('./images/Pearson Heat Map.svg', dpi=400, bbox_inches='tight', format='svg')

# ---------- 最大信息系数(MIC) ----------
# features = df.columns.values
# mine = MINE(alpha=0.6, c=15)
# mic = []
# for feature in features:
#     mine.compute_score(df[feature].values, df['FSC'].values)
#     print(mine.mic())
#     mic.append(mine.mic())
# print(mic)

# ---------- 展示测试集图像 ----------
# paths = []
# dates = []
# for file_name in ['133038_20131110', '147038_20141217', '150036_20170213', '146038_20180119', '152034_20181129', '149035_20201231']:
#     paths.append('IMGValidation/' + file_name + '.tif')
#     dates.append(file_name.split('_')[1])
# print(paths)
#
# imgs = []
# for i in range(len(paths)):
#     imgs.append(rioxarray.open_rasterio(paths[i]))
#
# fig, axes = plt.subplots(2, 3, figsize=(14, 8))
# for ax, img in zip(axes.flat, imgs):
#     im = ax.imshow(img[0], cmap='viridis')
# for ax, title in zip(axes.flat, dates):
#     ax.set_title(title)
# fig.colorbar(mappable=im, label='FSC', orientation='horizontal', ax=axes, fraction=0.05)
# # plt.savefig('./images/Dataset_Split.svg', dpi=150, bbox_inches='tight', format='svg')
# plt.show()

# ---------- 绘制MSE ----------
RF_MSE = pd.read_excel('result/RF.xlsx')[3]
XGB_MSE = pd.read_excel('result/XGB.xlsx')[3]
LGBM_MSE = pd.read_excel('result/LGBM.xlsx')[3]
NN_MSE = pd.read_excel('result/NN.xlsx')[3]
Blend_MSE = pd.read_excel('result/Blend.xlsx')[3]
Blend2_MSE = pd.read_excel('result/Blend-v2.xlsx')[3]
x = np.arange(1, 28)

MSE_fig = go.Figure(
    data=[
        go.Scatter(name='RF MSE', x=x, y=RF_MSE, textfont=dict(size=25),
                   mode='lines+markers', marker=dict(color='#00B050')),
        go.Scatter(name='XGB MSE', x=x, y=XGB_MSE, textfont=dict(size=25),
                   mode='lines+markers', marker=dict(color='#0068C9')),
        go.Scatter(name='LGBM MSE', x=x, y=LGBM_MSE, textfont=dict(size=25),
                   mode='lines+markers', marker=dict(color='#FF2B2B')),
        go.Scatter(name='NN MSE', x=x, y=NN_MSE, textfont=dict(size=25),
                   mode='lines+markers', marker=dict(color='#AB63FA')),
        go.Scatter(name='Blend-v1 MSE', x=x, y=Blend_MSE, textfont=dict(size=25),
                   mode='lines+markers', marker=dict(color='#FFA15A')),
        go.Scatter(name='Blend-v2 MSE', x=x, y=Blend2_MSE, textfont=dict(size=25),
                   mode='lines+markers', marker=dict(color='#19D3F3'))
    ]
)
MSE_fig.update_layout(
    autosize=False, width=1500, height=900,
    title=f'Predict MSE Loss',
    xaxis=dict(title='ID'),
    yaxis=dict(title='Loss', nticks=7, range=(0, 0.03)),
    # plot_bgcolor='#F6F6F6',
    showlegend=True
)
MSE_fig.show()
# pio.write_image(MSE_fig, f'images/Predict MSE Loss.svg')
