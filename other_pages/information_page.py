import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image
from minepy import MINE
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import rioxarray
import random
import os
import plotly.io as pio
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def load_css(css_file):
    with open(css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def app():
    st.write('# Information Page')
    load_css('style/style.css')
    lottie_coding = load_lottie('https://assets8.lottiefiles.com/private_files/lf30_w5u9xr3a.json')
    df = pd.read_csv('Data/test_data.csv')

    with st.container():
        st.write('---')
        image_column, text_column = st.columns((1, 3))
        with image_column:
            st.header('MODIS 数据集简介')
            st_lottie(lottie_coding, height=110, key="coding")
        with text_column:
            st.write(
                """
                雪是重要的水资源，冬季约占北半球总表面积的34%，融雪水是山区特别是干旱地区许多河流的主要水源。

                积雪有着特殊的物理性质，比如高反射率、低发射率和高潜热，所以积雪面积的变化强烈影响全球辐射能平衡。

                随着地理信息技术和遥感技术的不断发展，研究者们逐步开展了一系列积雪产品相关的研发。

                MODIS 积雪产品具有较高的时间分辨率，其空间分辨率为500m，近几年被广泛应用于积雪时空变化分析研究。

                数据集来自于 Google Earth Engine（GEE）平台，经过归一化处理后得到 TIFF 图像。
                """
            )

    with st.container():
        st.write('---')
        left_column, right_column = st.columns((1, 1))
        with left_column:
            st.header('训练集划分')
            st.write('##')
            st.write(
                """
                上述在 Google Earth Engine 平台获得的经过归一化处理的 TIFF 图像数据集一共有80张图片。

                我们将数据集划分成训练集和测试集（Test Data），其中，训练集有53张图像，测试集有27张图像。

                验证集从训练集中划分，`Train Data` 与 `Valid Data` 的比例为 `7 : 3`。
                
                即37张图像用来训练，16张图像用来验证。
                """
            )
        with right_column:
            # data_split = Image.open('images/Dataset_Split.png')
            # st.image(data_split, width=350)

            # dataset_split_fig = plt.figure(dpi=100, figsize=(5, 3))
            # plt.pie([7, 3], explode=[0.1, 0], labels=['Train data', 'Valid data'],
            #         colors=['lightcoral', 'lightskyblue'], autopct='%1.1f%%', shadow=True)
            # plt.title('MODIS 训练集划分')
            # # plt.legend()
            # st.pyplot(dataset_split_fig)

            dataset_split_fig = go.Figure(
                go.Pie(labels=['Train data', 'Valid data'], values=[7, 3],
                       textinfo='percent', hoverinfo='label+percent',
                       textfont=dict(size=15), pull=[0, 0.05],
                       title='MODIS 训练集划分', titlefont=dict(size=18),
                       marker=dict(colors=['#0068C9', '#83C9FF']))
            )
            dataset_split_fig.update_layout(
                autosize=False, width=600, height=450
            )
            st.plotly_chart(dataset_split_fig)
            pio.write_image(dataset_split_fig, 'images/Dataset_split.png')

    with st.container():
        st.write('---')
        st.header('数据集分析')
        st.write('##')
        st.subheader('数据展示')
        st.write('##')
        st.write(
            """
            本数据集选择七个通道的地表反射率、NDVI、NDSI、NDFSI、传感器方位角、传感器天顶角、太阳方位角、太阳天顶角、MCD12Q1 Landcover、
            海拔、坡度、坡向、地表覆盖类型 LCT、地表温度 LST 等与积雪面积有关的特征作为 FSC 反演的特征空间。
            
            我们对 `Valid Data` 进行数据分析。
            """
        )
        st.write('##')

        if st.button('View dataframe describe'):
            st.write('##')
            st.write(df.describe())

            # ---------- 数据分布直方图 ----------
            st.write('##')
            features1 = ['FSC', 'SR1', 'SR2', 'SR3', 'SR4', 'SR5', 'SR6', 'SR7', 'NDVI', 'NDSI', 'NDFSI']
            features2 = ['LCT', 'SensorZenith', 'SensorAzimuth', 'SolarZenith', 'SolarAzimuth', 'Dem', 'Slope', 'Aspect', 'LST', 'A2T', 'SC']
            colors = ['#66A4DF', '#B5DFFF', '#FF8080', '#FFCDCD', '#7FD0C4', '#B1F5C7', '#FFB766', '#FFE3A6', '#A78CD9', '#E6E9EF', '#0068C9']
            distribution_linechart_fig1 = go.Figure(
                data=[go.Histogram(name=feature, x=df.head(10000)[feature].values,
                                   marker=dict(color=color)) for feature, color in zip(features1, colors)],
            )
            distribution_linechart_fig2 = go.Figure(
                data=[go.Histogram(name=feature, x=df.head(10000)[feature].values,
                                   marker=dict(color=color)) for feature, color in zip(features2, colors)]
            )

            for fig in [distribution_linechart_fig1, distribution_linechart_fig2]:
                if fig == distribution_linechart_fig1:
                    fig.update_layout(title='Histogram of 10K Data Distribution')
                fig.update_layout(
                    barmode='overlay',  # 设置覆盖模式
                    autosize=False, width=1350, height=600,
                    xaxis=dict(title='Value'),
                    yaxis=dict(title='Count'),
                    showlegend=True,
                )
                fig.update_traces(opacity=0.6)  # 设置透明度

            st.plotly_chart(distribution_linechart_fig1)
            st.plotly_chart(distribution_linechart_fig2)
            # pio.write_image(distribution_linechart_fig1, 'images/Distribution_linechart1.svg')
            # pio.write_image(distribution_linechart_fig2, 'images/Distribution_linechart2.svg')
        st.write('##')

        if st.button('View some random data'):
            st.write('##')
            st.write(df.sample(10))

            st.write('##')
            st.bar_chart(df.sample(10))
            left_column, right_column = st.columns((4, 5))
            with right_column:
                st.caption('数据条形图')
            # st.write('##')
            # st.line_chart(df.sample(10))
            # left_column, right_column = st.columns((4, 5))
            # with right_column:
            #     st.caption('数据折线图')

        st.write('##')

        st.subheader('数据相关性分析')
        st.write('##')
        st.write(
            """
            数据的特征数量较多，可能存在与 FSC 值关联较小的特征或者数据存在错误的特征，因此可能需要对数据的特征进行提取，确定出较优的特征组合。
            
            确定数据多种特征之间的关系的方法有很多，比如散点图、相关系数、机器学习算法等：
            
             - 散点图可以直观地显示两个变量之间的分布情况，如果有多个变量，可以绘制散点矩阵；
             - 相关系数可以衡量两个变量同时变化的程度和方向的统计量，范围是 $-1\sim 1$，$1$ 表示完全正相关，$-1$ 表示完全负相关，即相关系数的绝对值越大表示两个变量之间的关系越紧密；
             - 机器学习算法可以使用不同的模型和参数来评估特征子集的效果，也可以检测特征之间的交互关系。
            
            我们通过计算各个特征与 FSC 值之间的皮尔逊（Pearson）相关系数来确定较优特征组合，此外通过绘制热力图用于直观地显示多个变量之间的相关系数。
            """
        )
        st.write('##')

        # ---------- 特征的标准差（离散度） ----------
        # 将标准差小于0.1的特征筛去后剩余：SR1, SR2, SR3, SR4, NDVI, NDSI, NDFSI, SensorAzimuth, SolarAzimuth,
        # Dem, Aspect, LST, SC, LCT
        std = df.std()
        features = std.index.values
        std_val = std.values[1:]  # 不算FSC
        # ---------- plotly ----------
        std_linechart_fig = go.Figure(
            data=[
                go.Bar(x=features[1:], y=std_val, textfont=dict(size=25),
                       marker=dict(color='#0068C9'))  # [1:]表示去掉FSC特征
            ]
        )
        std_linechart_fig.update_layout(
            autosize=False, width=1000, height=600,
            title='Standard Deviation',
            xaxis=dict(title='Feature'),
            yaxis=dict(title='Value'),
            showlegend=False
        )
        st.plotly_chart(std_linechart_fig)
        # pio.write_image(std_linechart_fig, 'images/Std_linechart.svg')
        st.write('##')

        # ---------- 皮尔逊相关系数折线图 ----------
        # 光谱信息和环境信息的阈值分别为0.4和0.2，进一步筛选特征后剩余：SR1, SR2, SR3, SR4, NDVI, NDSI, NDFSI,
        # Dem, LST, SC, LCT
        pearson = df.corr()
        pearson_fsc = pearson['FSC']
        pearson_fsc_val = pearson_fsc.values[1:]

        # ---------- matplotlib ----------
        # pearson_linechart_fig = plt.figure(dpi=300, figsize=(9.5, 4))
        # plt.plot(features, val, 'bo-', alpha=1, linewidth=1, label='Pearson Correlation Coefficient')
        # plt.legend()
        # plt.xlabel('Feature')
        # plt.ylabel('Value')
        # plt.ylim(-1, 1)
        # plt.xticks(rotation=60)
        # plt.axhline(0, color='gray')
        # st.pyplot(pearson_linechart_fig)

        # ---------- plotly ----------
        pearson_linechart_fig = go.Figure(
            data=[
                go.Scatter(name='Pearson Correlation Coefficient', x=features[1:], y=pearson_fsc_val,  # [1:]表示去掉FSC特征
                           textfont=dict(size=25), mode='lines+markers', marker=dict(color='#0068C9'))
            ]
        )
        pearson_linechart_fig.update_layout(
            autosize=False, width=1200, height=650,
            title='Pearson Correlation Coefficient Linechart',
            xaxis=dict(title='Feature'),
            yaxis=dict(title='Value', nticks=11, range=(-1, 1)),
            showlegend=True
        )
        st.plotly_chart(pearson_linechart_fig)
        # pio.write_image(pearson_linechart_fig, 'images/Pearson_linechart.svg')

        # ---------- 调用现成图 ----------
        # pearson_linechart = Image.open('images/Pearson Correlation Coefficient.png')
        # st.image(pearson_linechart, width=950)
        st.write('##')

        # ---------- 热力图 ----------

        # ---------- seaborn ----------
        # pearson_heatmap_fig = plt.figure(dpi=300, figsize=(12, 10))
        # sns.heatmap(df.corr(method='pearson'), linewidths=0.1, vmax=1.0, square=True, linecolor='white',
        #             annot=True, annot_kws={'size': 7})
        # plt.title('Pearson Heat Map')
        # st.pyplot(pearson_heatmap_fig)

        # ---------- plotly ----------
        pearson_heatmap_fig_colorscale = st.sidebar.selectbox(
            'Select Heat Colorscale',
            ('aggrnyl', 'agsunset', 'blues', 'brwnyl', 'burg', 'darkmint',
             'ice', 'oranges', 'pubu', 'rdbu', 'rdpu', 'reds'),
            index=2
        )
        pearson_heatmap_fig = go.Figure(
            data=[
                go.Heatmap(x=features, y=features, z=pearson.values, colorscale=pearson_heatmap_fig_colorscale)
            ]
        )
        pearson_heatmap_fig.update_layout(
            autosize=False, width=900, height=900,
            title='Pearson Correlation Coefficient Heatmap',
            # 保存本地时调整字体大小
            # xaxis=dict(title='Feature', titlefont=dict(size=10), tickfont=dict(size=8)),
            # yaxis=dict(title='Feature', titlefont=dict(size=10), tickfont=dict(size=8)),
            xaxis=dict(title='Feature'),
            yaxis=dict(title='Feature'),
            showlegend=True
        )
        st.plotly_chart(pearson_heatmap_fig)
        # pio.write_image(pearson_heatmap_fig, 'images/Pearson_heatmap.svg')

        # ---------- 调用现成图 ----------
        # pearson_heatmap = Image.open('images/Pearson Heat Map.png')
        # st.image(pearson_heatmap, width=950)
        st.write('##')

        st.subheader('图像展示')
        st.write('##')

        # 展示同一张图不同波段
        if st.button('Randomly show different bands of same images'):
            st.write('##')
            band1, band2 = random.randint(0, 21), random.randint(0, 21)
            while band1 == band2:
                band2 = random.randint(0, 21)
            same_img_path = 'IMGTrain/130036_20150111.tif'
            same_img = rioxarray.open_rasterio(same_img_path)
            left_column, right_column = st.columns((2, 1))
            with left_column:
                tiff_img_fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                for ax, img in zip(axes.flat, [same_img[band1], same_img[band2]]):
                    im = ax.imshow(img, cmap='viridis')
                for ax, title in zip(axes.flat, [f'Image 1 Band {band1}', f'Image 1 Band {band2}']):
                    ax.set_title(title)
                tiff_img_fig.colorbar(mappable=im, label='FSC', orientation='horizontal', ax=axes, fraction=0.05)
                st.pyplot(tiff_img_fig)
        st.write('##')

        # 展示不同图像的相同波段
        if st.button('Randomly show different images of same bands'):
            st.write('##')
            img1, img2 = random.randint(0, 52), random.randint(0, 52)
            while img1 == img2:
                img2 = random.randint(0, 52)
            train_data_path = 'IMGTrain'
            diff_img_path = os.listdir(train_data_path)
            diff_img_path1, diff_img_path2 = os.path.join(train_data_path, diff_img_path[img1]), os.path.join(train_data_path, diff_img_path[img2])
            diff_img1, diff_img2 = rioxarray.open_rasterio(diff_img_path1), rioxarray.open_rasterio(diff_img_path2)
            left_column, right_column = st.columns((2, 1))
            with left_column:
                tiff_img_fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                for ax, img in zip(axes.flat, [diff_img1[0], diff_img2[0]]):
                    im = ax.imshow(img, cmap='viridis')
                for ax, title in zip(axes.flat, [f'Image {img1} Band 0', f'Image {img2} Band 0']):
                    ax.set_title(title)
                tiff_img_fig.colorbar(mappable=im, label='FSC', orientation='horizontal', ax=axes, fraction=0.05)
                st.pyplot(tiff_img_fig)
        st.write('##')

        # ---------- 展示以上两类图 ----------
        if st.button('Show original validation image'):
            st.write('##')

            original_img1_path = 'IMGTrain/130036_20150111.tif'
            original_img1 = rioxarray.open_rasterio(original_img1_path)

            original_img_path = 'IMGTrain'
            original_img_path_list = os.listdir(original_img_path)
            original_img2_path = os.path.join(original_img_path, original_img_path_list[12])
            original_img3_path = os.path.join(original_img_path, original_img_path_list[52])
            original_img2 =  rioxarray.open_rasterio(original_img2_path)
            original_img3 =  rioxarray.open_rasterio(original_img3_path)

            left_column, right_column = st.columns((2, 1))
            with left_column:
                tiff_img_fig, axes = plt.subplots(1, 4, figsize=(12, 4))
                for ax, img in zip(axes.flat, [original_img1[15], original_img1[4], original_img2[0], original_img3[0]]):
                    im = ax.imshow(img, cmap='viridis')
                for ax, title in zip(axes.flat, [f'Image 1 Band 15', f'Image 1 Band 4',
                                                 f'Image 12 Band 0', f'Image 52 Band 0']):
                    ax.set_title(title)
                tiff_img_fig.colorbar(mappable=im, label='FSC', orientation='horizontal', ax=axes, fraction=0.08)
                st.pyplot(tiff_img_fig)
                # plt.savefig('images/Original_tiff_image.svg', dpi=150, bbox_inches='tight', format='svg')
        st.write('##')

        # ---------- 展示Test数据 ----------
        if st.button('Show original test image'):
            st.write('##')

            paths = []
            dates = []
            for file_name in ['133038_20131110', '147038_20141217', '150036_20170213', '146038_20180119', '152034_20181129',
                              '149035_20201231']:
                paths.append('IMGValidation/' + file_name + '.tif')
                dates.append(file_name.split('_')[1])

            imgs = []
            for i in range(len(paths)):
                imgs.append(rioxarray.open_rasterio(paths[i]))

            left_column, right_column = st.columns((2, 1))
            with left_column:
                fig, axes = plt.subplots(2, 3, figsize=(13, 8))
                for ax, img in zip(axes.flat, imgs):
                    im = ax.imshow(img[0], cmap='viridis')
                for ax, title in zip(axes.flat, dates):
                    ax.set_title(title)
                fig.colorbar(mappable=im, label='FSC', orientation='horizontal', ax=axes, fraction=0.04)
                # plt.savefig('./images/Dataset_Split.svg', dpi=150, bbox_inches='tight', format='svg')
                st.pyplot(fig)
