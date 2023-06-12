import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import rasterio
import rioxarray
import tqdm
import os
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
import plotly.io as pio
from sklearn.metrics import mean_squared_error, mean_absolute_error
from train_nn import MLP
import torch
import torch.nn as nn
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data(path):
    df = pd.read_csv(path)
    data_sets = pd.DataFrame(df, dtype=np.float32)
    feature_data = data_sets.drop(['FSC'], axis=1)
    # feature_data.drop(columns=['NDVI', 'NDSI', 'NDFSI', 'SC'], inplace=True)
    label_data = data_sets['FSC']

    return feature_data, label_data

def read_val_image(img_path):
    img_data = rasterio.open(img_path).read()
    img_shape = np.shape(img_data)
    band_num, height_num, width_num = img_shape

    img_data_list, row_col = [], []
    for i in tqdm.trange(height_num):
        for j in range(width_num):
            temp = img_data[::, i, j]
            if np.array(np.isnan(temp), dtype=np.int8).sum() > 0:
                continue
            else:
                img_data_list.append(temp.tolist())
                row_col.append([i, j])

    img_arr = np.array(img_data_list)
    labels = img_arr[:, 0]
    feature_data = img_arr[:, 1:]
    # feature_name = ['SR1', 'SR2', 'SR3', 'SR4', 'SR5', 'SR6', 'SR7', 'NDVI', 'NDSI', 'NDFSI',
    #                 'SensorZenith', 'SensorAzimuth', 'SolarZenith', 'SolarAzimuth',
    #                 'Dem', 'Slope', 'Aspect', 'LST', 'A2T', 'SC', 'LCT']
    # feature_data = pd.DataFrame(feature_data, columns=feature_name)
    # feature_data.drop(columns=['NDVI', 'NDSI', 'NDFSI', 'SC'], inplace=True)
    rows_cols = np.array(row_col)

    return feature_data, labels, rows_cols, f'({img_shape[0]}, {img_shape[1]}, {img_shape[2]})'

def export_pred_img(vl_path, rows_cols, val_pred, out_path):
    img_data = rasterio.open(vl_path).read()

    for a in range(len(rows_cols)):
        rc = rows_cols[a]
        img_data[0][rc[0]][rc[1]] = val_pred[a]

    print(np.shape(img_data))
    x_size, y_size = np.shape(img_data[0])
    with rasterio.open(out_path, 'w',
                       driver='GTiff',
                       width=y_size,
                       height=x_size,
                       count=22,
                       crs=rasterio.open(vl_path).crs,
                       transform=rasterio.open(vl_path).transform,
                       dtype=rasterio.float32,
                       nodata=255) as dt:
        dt.write(img_data.astype(rasterio.float32))
    dt.close()

def predict(model):
    val_p = './IMGValidation'
    out_p = './IMGPred'

    predictions, MSE, RMSE, MAE = [], [], [], []
    val_path_list = os.listdir(val_p)

    progress = st.progress(0)
    # for i in range(0, 5):
    for i in range(0, len(val_path_list)):
        val_path = os.path.join(val_p, val_path_list[i])
        out_name = val_path_list[i].split('.')[0] + '_pred.tif'
        out_path = os.path.join(out_p, out_name)

        val_data, val_label, rcs, img_shape = read_val_image(val_path)
        label_hat = model.predict(val_data)
        MSE_val = mean_squared_error(val_label, label_hat)
        RMSE_val = MSE_val**0.5
        MAE_val = mean_absolute_error(val_label, label_hat)

        export_pred_img(vl_path=val_path, rows_cols=rcs, val_pred=label_hat, out_path=out_path)

        res = [val_path_list[i], img_shape, val_label.sum(), label_hat.sum(), MSE_val, RMSE_val, MAE_val]
        predictions.append(res)
        MSE.append(MSE_val)
        RMSE.append(RMSE_val)
        MAE.append(MAE_val)

        progress.progress((i + 1) / len(val_path_list))

    predictions = np.array(predictions)
    attribute = ['Image Name', 'Shape', 'FSC Truth', 'FSC Pred', 'MSE', 'RMSE', 'MAE']
    df = pd.DataFrame(predictions, columns=attribute)

    return df, MSE, RMSE, MAE

# 使用机器学习模型预测20181129
# def predict(model):
#     val_p = './IMGPred_20181129/152034_20181129_gt.tif'
#     out_p = './IMGPred_20181129/152034_20181129'
#
#     predictions, MSE, RMSE, MAE = [], [], [], []
#     out_path = out_p + '_rf.tif'
#     val_data, val_label, rcs, img_shape = read_val_image(val_p)
#     label_hat = model.predict(val_data)
#     MSE_val = mean_squared_error(val_label, label_hat)
#     RMSE_val = MSE_val**0.5
#     MAE_val = mean_absolute_error(val_label, label_hat)
#
#     export_pred_img(vl_path=val_p, rows_cols=rcs, val_pred=label_hat, out_path=out_path)
#
#     res = ['152034_20181129.tif', img_shape, val_label.sum(), label_hat.sum(), MSE_val, RMSE_val, MAE_val]
#     predictions.append(res)
#     MSE.append(MSE_val)
#     RMSE.append(RMSE_val)
#     MAE.append(MAE_val)
#
#     predictions = np.array(predictions)
#     attribute = ['Image Name', 'Shape', 'FSC Truth', 'FSC Pred', 'MSE', 'RMSE', 'MAE']
#     df = pd.DataFrame(predictions, columns=attribute)
#
#     return df, MSE, RMSE, MAE

# 使用神经网络预测20181129
def nn_predict(model, loss_function, device):
    val_p = './IMGPred_20181129/152034_20181129_gt.tif'
    out_p = './IMGPred_20181129/152034_20181129'

    predictions, MSE, RMSE, MAE = [], [], [], []
    out_path = out_p + '_nn.tif'
    val_data, val_label, rcs, img_shape = read_val_image(val_p)
    val_data_tensor = torch.tensor(np.array(val_data), dtype=torch.float32).to(device)
    val_label_tensor = torch.tensor(np.array(val_label), dtype=torch.float32).reshape(-1, 1).to(device)
    with torch.no_grad():
        label_hat = model(val_data_tensor)
    MSE_val = loss_function(label_hat, val_label_tensor).item()
    RMSE_val = MSE_val**0.5
    MAE_val = mean_absolute_error(val_label, label_hat.cpu().numpy())

    export_pred_img(vl_path=val_p, rows_cols=rcs, val_pred=label_hat, out_path=out_path)

    res = ['152034_20181129.tif', img_shape, val_label.sum(), label_hat.sum().cpu().numpy(), MSE_val, RMSE_val, MAE_val]
    predictions.append(res)
    MSE.append(MSE_val)
    RMSE.append(RMSE_val)
    MAE.append(MAE_val)

    predictions = np.array(predictions)
    attribute = ['Image Name', 'Shape', 'FSC Truth', 'FSC Pred', 'MSE', 'RMSE', 'MAE']
    df = pd.DataFrame(predictions, columns=attribute)

    return df, MSE, RMSE, MAE

def blend_models_predict(X, lgb_weight, xgb_weight, rf_weight, lgb_reg, xgb_reg, rf_reg):
    return (lgb_weight * lgb_reg.predict(X) + xgb_weight * xgb_reg.predict(X) + rf_weight * rf_reg.predict(X))

def integration_predict(lgb_weight, xgb_weight, rf_weight, lgb_reg, xgb_reg, rf_reg):
    val_p = './IMGValidation'
    out_p = './IMGPred'

    predictions, MSE, RMSE, MAE = [], [], [], []
    val_path_list = os.listdir(val_p)

    progress = st.progress(0)
    for i in range(0, len(val_path_list)):
        val_path = os.path.join(val_p, val_path_list[i])
        out_name = val_path_list[i].split('.')[0] + '_pred.tif'
        out_path = os.path.join(out_p, out_name)

        val_data, val_label, rcs, img_shape = read_val_image(val_path)
        label_hat = blend_models_predict(val_data, lgb_weight, xgb_weight, rf_weight, lgb_reg, xgb_reg, rf_reg)
        MSE_val = mean_squared_error(val_label, label_hat)
        RMSE_val = MSE_val ** 0.5
        MAE_val = mean_absolute_error(val_label, label_hat)

        export_pred_img(vl_path=val_path, rows_cols=rcs, val_pred=label_hat, out_path=out_path)

        res = [val_path_list[i], img_shape, val_label.sum(), label_hat.sum(), MSE_val, RMSE_val, MAE_val]
        predictions.append(res)
        MSE.append(MSE_val)
        RMSE.append(RMSE_val)
        MAE.append(MAE_val)

        progress.progress((i + 1) / len(val_path_list))

    predictions = np.array(predictions)
    attribute = ['Image Name', 'Shape', 'FSC Truth', 'FSC Pred', 'MSE', 'RMSE', 'MAE']
    df = pd.DataFrame(predictions, columns=attribute)

    return df, MSE, RMSE, MAE

# 未完成
def add_parameters_ui(model_name):
    params = dict()
    if model_name == 'LGBMRegressor':
        num_leaves = st.sidebar.slider('num_leaves', 1, 30, 15, 1)
        learning_rate = st.sidebar.slider('learning_rate', 0.01, 0.3, 0.1, 0.01)
        n_estimators = st.sidebar.slider('n_estimators', 100, 1500, 800, 10)
        max_bin = st.sidebar.slider('max_bin', 10, 100, 55, 1)
        bagging_fraction = st.sidebar.slider('bagging_fraction', 0.1, 1.0, 0.8, 0.01)
        feature_fraction = st.sidebar.slider('feature_fraction', 0.1, 1.0, 0.71, 0.01)
        min_data_in_leaf = st.sidebar.slider('min_data_in_leaf', 1, 20, 6, 1)
        min_sum_hessian_in_leaf = st.sidebar.slider('min_sum_hessian_in_leaf', 1, 20, 11, 1)
        params['num_leaves'] = num_leaves
        params['learning_rate'] = learning_rate
        params['n_estimators'] = n_estimators
        params['max_bin'] = max_bin
        params['bagging_fraction'] = bagging_fraction
        params['feature_fraction'] = feature_fraction
        params['min_data_in_leaf'] = min_data_in_leaf
        params['min_sum_hessian_in_leaf'] = min_sum_hessian_in_leaf
    elif model_name == 'XGBRegressor':
        colsample_bytree = st.sidebar.slider('colsample_bytree', 1, 30, 15, 1)
        gamma = st.sidebar.slider('gamma', 0.01, 1.0, 0.04, 0.01)
        learning_rate = st.sidebar.slider('learning_rate', 0.01, 0.3, 0.13, 0.01)
        max_depth = st.sidebar.slider('max_depth', 1, 20, 5, 1)
        min_child_weight = st.sidebar.slider('min_child_weight', 1.0, 10.0, 2.78, 0.01)
        n_estimators = st.sidebar.slider('n_estimators', 100, 1500, 800, 10)
        reg_alpha = st.sidebar.slider('reg_alpha', 0.1, 1.0, 0.96, 0.01)
        reg_lambda = st.sidebar.slider('reg_lambda', 0.1, 1.0, 0.95, 0.01)
        params['colsample_bytree'] = colsample_bytree
        params['gamma'] = gamma
        params['learning_rate'] = learning_rate
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight
        params['n_estimators'] = n_estimators
        params['reg_alpha'] = reg_alpha
        params['reg_lambda'] = reg_lambda
    elif model_name == 'GradientBoostingRegressor':
        pass
    elif model_name == 'RandomForestRegressor':
        pass
    return params

# 未完成
def get_model(model_name, params):
    model = None
    if model_name == 'LGBMRegressor':
        model = lgb.LGBMRegressor(objective='regression', num_leaves=params['num_leaves'],
                                  learning_rate=params['learning_rate'], n_estimators=params['n_estimators'],
                                  max_bin=params['max_bin'], bagging_fraction=params['bagging_fraction'],
                                  bagging_freq=5, feature_fraction=params['feature_fraction'],
                                  feature_fraction_seed=9, bagging_seed=9, min_data_in_leaf=params['min_data_in_leaf'],
                                  min_sum_hessian_in_leaf=params['min_sum_hessian_in_leaf'])
    elif model_name == 'XGBRegressor':
        pass
    elif model_name == 'GradientBoostingRegressor':
        pass
    elif model_name == 'RandomForestRegressor':
        pass
    return model

def get_model_weight():
    model_weight = dict()
    LGBMRegressor = st.sidebar.slider('LGBMRegressor Weight', 0.1, 1.0, 0.4, 0.1)
    XGBRegressor = st.sidebar.slider('XGBRegressor Weight', 0.1, 1.0, 0.3, 0.1)
    RandomForestRegressor = st.sidebar.slider('RandomForestRegressor Weight', 0.1, 1.0, 0.3, 0.1)
    model_weight['LGBMRegressor'] = LGBMRegressor
    model_weight['XGBRegressor'] = XGBRegressor
    model_weight['RandomForestRegressor'] = RandomForestRegressor
    return model_weight

def app():
    st.write('# Machine Learning Page')

    with st.container():
        st.write('---')
        st.header('Pretrained Model')
        st.write('##')

        st.sidebar.subheader('Pretrained Model')
        pretrained_model_name = st.sidebar.selectbox(
            'Select Pretrained Model',
            ('LGBMRegressor', 'XGBRegressor', 'GradientBoostingRegressor', 'RandomForestRegressor')
        )

        pretrained_model = joblib.load(f"./model/{pretrained_model_name}.pkl")

        st.subheader(f'{type(pretrained_model).__name__}')
        st.write('##')

        if st.button(f'Use pretrained {type(pretrained_model).__name__} predict'):
            st.write('##')

            # 机器学习模型预测
            df_pred, MSE_pred, RMSE_pred, MAE_pred = predict(pretrained_model)

            # 神经网络预测
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # net = MLP()
            # net.to(device)
            # net.load_state_dict(torch.load('./save/DL_no_feature_select.params'))
            # loss_function = nn.MSELoss(reduction='mean')
            # loss_function.to(device)
            # net.eval()
            # df_pred, MSE_pred, RMSE_pred, MAE_pred = nn_predict(net, loss_function, device)

            st.write('Partial predict result：')
            st.write('##')
            st.write(df_pred.head())
            st.write('##')

            # st.write('Loss：')
            # left_column, right_column = st.columns((5, 2))
            # with left_column:
            #     pred_linechart_fig = plt.figure(dpi=300, figsize=(6, 4))
            #     plt.plot([x for x in range(len(MSE_pred))], MSE_pred, 'o-', alpha=1, linewidth=0.5, markersize=2, label='MSE')
            #     plt.plot([x for x in range(len(RMSE_pred))], RMSE_pred, 'o-', alpha=1, linewidth=0.5, markersize=2, label='RMSE')
            #     plt.plot([x for x in range(len(MAE_pred))], MAE_pred, 'o-', alpha=1, linewidth=0.5, markersize=2, label='MAE')
            #     plt.legend(fontsize=7)
            #     plt.xlabel('Test Image', fontsize=10)
            #     plt.ylabel('Error Value', fontsize=10)
            #     plt.xticks(fontsize=6)
            #     plt.yticks(fontsize=6)
            #     plt.ylim(0, 0.25)
            #     st.pyplot(pred_linechart_fig)

            x = [i for i in range(len(MSE_pred))]
            loss_linechart_fig = go.Figure(
                data=[
                    go.Scatter(name='MSE Loss', x=x, y=MSE_pred, textfont=dict(size=25),
                               mode='lines+markers', marker=dict(color='#0068C9')),
                    go.Scatter(name='RMSE Loss', x=x, y=RMSE_pred, textfont=dict(size=25),
                               mode='lines+markers', marker=dict(color='#83C9FF')),
                    go.Scatter(name='MAE Loss', x=x, y=MAE_pred, textfont=dict(size=25),
                               mode='lines+markers', marker=dict(color='#FF2B2B'))
                ]
            )
            loss_linechart_fig.update_layout(
                autosize=False, width=1200, height=650,
                title=f'{pretrained_model_name} Predict Loss',
                xaxis=dict(title='ID'),
                yaxis=dict(title='Loss', nticks=12, range=(0, 0.22)),
                showlegend=True
            )
            st.plotly_chart(loss_linechart_fig)
            # pio.write_image(loss_linechart_fig, f'images/{pretrained_model_name}_loss_linechart.svg')
            st.write('##')

            met1, met2, met3 = st.columns(3)
            met1.metric('MSE Loss', f'{sum(MSE_pred) / len(MSE_pred):.4f}')
            met2.metric('RMSE Loss', f'{sum(RMSE_pred) / len(RMSE_pred):.4f}')
            met3.metric('MAE Loss', f'{sum(MAE_pred) / len(MAE_pred):.4f}')
        st.write('##')

    with st.container():
        st.write('---')
        st.header('FSC 反演结果')
        st.write('##')

        val_path = 'IMGValidation'
        pred_path = 'IMGPred'
        val_all_img_path = os.listdir(val_path)
        dates = []
        dates_to_names = {}
        for i in range(0, len(val_all_img_path)):
            date = val_all_img_path[i].split('.')[0].split('_')[1]
            dates.append(date)
            dates_to_names[date] = val_all_img_path[i]
        dates.sort()
        # dates_to_names = dict(sorted(dates_to_names.items(), key=lambda x: x[0], reverse=False))

        st.sidebar.write('---')
        st.sidebar.subheader('Predict')
        date = st.sidebar.selectbox(
            'Select Date',
            dates
        )

        if st.button('Show FSC prediction result (select date first)'):
            st.write('##')
            val_img_path = os.path.join(val_path, dates_to_names[date])
            pred_img_path = os.path.join(pred_path, dates_to_names[date].split('.')[0] + '_pred.tif')
            # st.write(pred_img_path)

            if not os.path.exists(pred_img_path):
                st.warning('You have not made a prediction!')

            else:
                val_img = rioxarray.open_rasterio(val_img_path)
                pred_img = rioxarray.open_rasterio(pred_img_path)
                left_column, right_column = st.columns((2, 1))
                with left_column:
                    # tiff_img_fig = plt.figure(dpi=350, figsize=(12, 4))
                    # plt.subplots_adjust(hspace=0.2, wspace=0.5)
                    # plt.subplot(1, 2, 1)
                    # # val_img[0].plot(cmap='terrain')
                    # plt.imshow(val_img[0], cmap='viridis')
                    # plt.subplot(1, 2, 2)
                    # # pred_img[0].plot(cmap='terrain')
                    # plt.imshow(pred_img[0], cmap='viridis')
                    # plt.colorbar(label='FSC', orientation='horizontal')

                    tiff_img_fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    for ax, img in zip(axes.flat, [val_img[0], pred_img[0]]):
                        im = ax.imshow(img, cmap='viridis')
                    for ax, title in zip(axes.flat, ['Truth Image', 'Pred Image']):
                        ax.set_title(title)
                    tiff_img_fig.colorbar(mappable=im, label='FSC', orientation='horizontal', ax=axes, fraction=0.05)
                    st.pyplot(tiff_img_fig)
        st.write('##')

        if st.button('Show 20181129 FSC prediction result'):
            st.write('##')
            fsc_paths = []
            fsc_names = []
            for model_name in ['gt', 'rf', 'xgb', 'lgbm', 'blend', 'nn']:
                fsc_paths.append(f'IMGPred_20181129/152034_20181129_{model_name}.tif')
                fsc_names.append(f'FSC-{model_name}'.upper())
            fsc_imgs = []
            for i in range(len(fsc_paths)):
                fsc_imgs.append(rioxarray.open_rasterio(fsc_paths[i]))

            left_column, right_column = st.columns((2, 1))
            with left_column:
                fig, axes = plt.subplots(2, 3, figsize=(13, 8))
                for ax, img in zip(axes.flat, fsc_imgs):
                    im = ax.imshow(img[0], cmap='viridis')
                for ax, title in zip(axes.flat, fsc_names):
                    ax.set_title(title)
                fig.colorbar(mappable=im, label='FSC', orientation='horizontal', ax=axes, fraction=0.04)
                # plt.savefig('./images/Dataset_Split.svg', dpi=150, bbox_inches='tight', format='svg')
                st.pyplot(fig)
        st.write('##')

    with st.container():
        st.write('---')
        st.header('Train ML Model')
        st.write('##')

        st.sidebar.write('---')
        st.sidebar.subheader('Train ML Model')
        model_name = st.sidebar.selectbox(
            'Select ML Model',
            ('LGBMRegressor', 'XGBRegressor', 'GradientBoostingRegressor', 'RandomForestRegressor')
        )
        params = add_parameters_ui(model_name)

        st.subheader(f'{model_name}')
        st.write('##')

        if st.button(f'Train {model_name}'):
            st.write('##')
            model = get_model(model_name, params)

            train_feature, train_label = load_data('Data/train_data.csv')
            test_feature, test_label = load_data('Data/test_data.csv')

            with st.spinner('Wating...'):
                model.fit(train_feature, train_label)
                label_pred = model.predict(test_feature)
                MSE_ts = mean_squared_error(test_label, label_pred)
            st.write(f'**{model_name} Pred MSE in Valid Data: {MSE_ts:.4f}**')
            st.write('##')

            st.info('You can click the button below to save the trained model!')
            if st.button('Save Model'):
                joblib.dump(model, f'model/{model_name}.pkl')
                st.success('Save successfully!')
        st.write('##')

    with st.container():
        st.write('---')
        st.header('Integration Model')
        st.write('##')

        st.sidebar.write('---')
        st.sidebar.subheader('Integration Model')

        model_weight = get_model_weight()
        if st.button('Use integration model predict'):
            st.write('##')
            lgb_reg = joblib.load("./model/LGBMRegressor.pkl")
            xgb_reg = joblib.load("./model/XGBRegressor.pkl")
            rf_reg = joblib.load("./model/RandomForestRegressor.pkl")
            df_pred, MSE_pred, RMSE_pred, MAE_pred = integration_predict(model_weight['LGBMRegressor'],
                                                                         model_weight['XGBRegressor'],
                                                                         model_weight['RandomForestRegressor'],
                                                                         lgb_reg, xgb_reg, rf_reg)
            st.write('Partial predict result：')
            st.write('##')
            st.write(df_pred.head())
            st.write('##')

            x = [i for i in range(len(MSE_pred))]
            integration_loss_linechart_fig = go.Figure(
                data=[
                    go.Scatter(name='MSE Loss', x=x, y=MSE_pred, textfont=dict(size=25),
                               mode='lines+markers', marker=dict(color='#0068C9')),
                    go.Scatter(name='RMSE Loss', x=x, y=RMSE_pred, textfont=dict(size=25),
                               mode='lines+markers', marker=dict(color='#83C9FF')),
                    go.Scatter(name='MAE Loss', x=x, y=MAE_pred, textfont=dict(size=25),
                               mode='lines+markers', marker=dict(color='#FF2B2B'))
                ]
            )
            integration_loss_linechart_fig.update_layout(
                autosize=False, width=1200, height=650,
                title='Integration Model Predict Loss',
                xaxis=dict(title='ID'),
                yaxis=dict(title='Loss', nticks=12, range=(0, 0.22)),
                showlegend=True
            )
            st.plotly_chart(integration_loss_linechart_fig)
            pio.write_image(integration_loss_linechart_fig, f'images/Integration_loss_linechart.svg')
        st.write('##')
