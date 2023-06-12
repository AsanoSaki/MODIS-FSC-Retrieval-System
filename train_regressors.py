from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mlxtend.regressor import StackingCVRegressor
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
import os
import rasterio
import tqdm
import joblib
import pickle
import torch
import torch.nn as nn
from train_nn import MLP

# ---------- Load NN ----------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = MLP()
net.to(device)
net.load_state_dict(torch.load('./save/DL_no_feature_select.params'))

loss_function = nn.MSELoss(reduction='mean')
loss_function.to(device)

net.eval()

def load_data(path):
    df = pd.read_csv(path)
    data_sets = pd.DataFrame(df, dtype=np.float32)
    feature_data = data_sets.drop(['FSC'], axis=1)
    # feature_data.drop(columns=['SensorZenith',
    #                            'SolarZenith', 'Slope', 'Aspect', 'A2T'], inplace=True)
    label_data = data_sets['FSC']

    return feature_data, label_data

train_feature, train_label = load_data('./Data/train_data.csv')
test_feature, test_label = load_data('./Data/valid_data.csv')


# ---------- Train ----------

# 特征优选前
lgb_reg = lgb.LGBMRegressor(objective='regression', num_leaves=15,
                            learning_rate=0.1, n_estimators=800,
                            max_bin=55, bagging_fraction=0.8,
                            bagging_freq=5, feature_fraction=0.713,
                            feature_fraction_seed=9, bagging_seed=9,
                            min_data_in_leaf=6, min_sum_hessian_in_leaf=11)  # 0.0167

xgb_reg = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                           learning_rate=0.13, max_depth=5,
                           min_child_weight=2.7817, n_estimators=800,
                           reg_alpha=0.9640, reg_lambda=0.9571,
                           subsample=0.5213, random_state=7, nthread=-1)  # 0.0170

GBoost = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.1,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=5)  # 0.0172

rf_reg = RandomForestRegressor(n_estimators=500, max_features=5, min_samples_leaf=5,
                                   oob_score=True, random_state=0, verbose=0, n_jobs=-1)  # 0.0166

stack_gen = StackingCVRegressor(regressors=(lgb_reg, xgb_reg, rf_reg),
                                meta_regressor=lgb_reg,
                                use_features_in_secondary=True)  # 0.0165

# 特征优选后
# lgb_reg = lgb.LGBMRegressor(objective='regression', num_leaves=14,
#                             learning_rate=0.1, n_estimators=600,
#                             max_bin=55, bagging_fraction=0.8,
#                             bagging_freq=5, feature_fraction=0.763,
#                             feature_fraction_seed=9, bagging_seed=9,
#                             min_data_in_leaf=6, min_sum_hessian_in_leaf=10)  # 0.0183

# lgb_reg.fit(train_feature, train_label)
# label_pred = lgb_reg.predict(test_feature)
# MSE_ts = mean_squared_error(test_label, label_pred)
# RMSE_ts = np.sqrt(MSE_ts)
# MAE_ts = mean_absolute_error(test_label, label_pred)
# print("\nLGBMRegressor Pred MSE: {:.4f}\n".format(MSE_ts))  # LGBMRegressor Pred MSE: 0.0119
# print("\nLGBMRegressor Pred RMSE: {:.4f}\n".format(RMSE_ts))  # LGBMRegressor Pred RMSE: 0.1089
# print("\nLGBMRegressor Pred MAE: {:.4f}\n".format(MAE_ts))  # LGBMRegressor Pred MAE: 0.0597
# joblib.dump(lgb_reg, "./model/LGBMRegressor.pkl")

# xgb_reg.fit(train_feature, train_label)
# label_pred = xgb_reg.predict(test_feature)
# MSE_ts = mean_squared_error(test_label, label_pred)
# RMSE_ts = np.sqrt(MSE_ts)
# MAE_ts = mean_absolute_error(test_label, label_pred)
# print("\nXGBRegressor Pred MSE: {:.4f}\n".format(MSE_ts))  # XGBRegressor Pred MSE: 0.0115
# print("\nXGBRegressor Pred RMSE: {:.4f}\n".format(RMSE_ts))  # XGBRegressor Pred RMSE: 0.1071
# print("\nXGBRegressor Pred MAE: {:.4f}\n".format(MAE_ts))  # XGBRegressor Pred MAE: 0.0596
# joblib.dump(xgb_reg, "./model/XGBRegressor.pkl")

# GBoost.fit(train_feature, train_label)
# label_pred = GBoost.predict(test_feature)
# MSE_ts = mean_squared_error(test_label, label_pred)
# print("\GradientBoostingRegressor Pred MSE: {:.4f}\n".format(MSE_ts))

# rf_reg.fit(train_feature, train_label)
# label_pred = rf_reg.predict(test_feature)
# MSE_ts = mean_squared_error(test_label, label_pred)
# RMSE_ts = np.sqrt(MSE_ts)
# MAE_ts = mean_absolute_error(test_label, label_pred)
# print("\nRandomForestRegressor Pred MSE: {:.4f}\n".format(MSE_ts))  # RandomForestRegressor Pred MSE: 0.0106
# print("\nRandomForestRegressor Pred RMSE: {:.4f}\n".format(RMSE_ts))  # RandomForestRegressor Pred RMSE: 0.1028
# print("\nRandomForestRegressor Pred MAE: {:.4f}\n".format(MAE_ts))  # RandomForestRegressor Pred MAE: 0.0533
# joblib.dump(rf_reg, "./model/RandomForestRegressor.pkl")

# stack_gen.fit(train_feature, train_label)  # StackingCVRegressor: MSE 0.0166, RMSE 0.1288, MAE 0.0755 (测试集结果)
# label_pred = stack_gen.predict(test_feature)
# MSE_ts = mean_squared_error(test_label, label_pred)
# RMSE_ts = np.sqrt(MSE_ts)
# MAE_ts = mean_absolute_error(test_label, label_pred)
# print("\nStackingCVRegressor Pred MSE: {:.4f}\n".format(MSE_ts))  # StackingCVRegressor Pred MSE: 0.0093
# print("\nStackingCVRegressor Pred RMSE: {:.4f}\n".format(RMSE_ts))  # StackingCVRegressor Pred RMSE: 0.0963
# print("\nStackingCVRegressor Pred MAE: {:.4f}\n".format(MAE_ts))  # StackingCVRegressor Pred MAE: 0.0485
# with open('./model/StackingCV.pickle', 'wb') as f:
#     pickle.dump(stack_gen, f)  # 出错：MemoryError
# joblib.dump(stack_gen, "./model/StackingCV.pkl")


# ---------- Test ----------

lgb_reg = joblib.load("./model/LGBMRegressor.pkl")  # LGBMRegressor: MSE 0.0168, RMSE 0.1296, MAE 0.0783
xgb_reg = joblib.load("./model/XGBRegressor.pkl")  # XGBRegressor: MSE 0.0172, RMSE 0.1311, MAE 0.0816
rf_reg = joblib.load("./model/RandomForestRegressor.pkl")  # RandomForestRegressor: MSE 0.0166, RMSE 0.1288, MAE 0.0766
# stack_gen = joblib.load("./model/StackingCV.pkl")

# with open('./model/StackingCV.pickle', 'rb') as f:
#     stack_gen = pickle.load(f)

# label_pred = stack_gen.predict(test_feature)
# MSE_ts = mean_squared_error(test_label, label_pred)
# RMSE_ts = np.sqrt(MSE_ts)
# MAE_ts = mean_absolute_error(test_label, label_pred)
# print("\nStackingCVRegressor Pred MSE: {:.4f}\n".format(MSE_ts))  # StackingCVRegressor Pred MSE: 0.0166
# print("\nStackingCVRegressor Pred RMSE: {:.4f}\n".format(RMSE_ts))  # StackingCVRegressor Pred RMSE: 0.1288
# print("\nStackingCVRegressor Pred MAE: {:.4f}\n".format(MAE_ts))  # StackingCVRegressor Pred MAE: 0.0755

def blend_models_predict(test_feature=None, test_label=None, have_best=False):  # BlendModels: MSE 0.0159, RMSE 0.1262, MAE 0.0744 (测试集结果) 权重: 2 1 3 4
    with torch.no_grad():
        test_tensor = torch.tensor(np.array(test_feature), dtype=torch.float32).to(device)
        nn_hat = net(test_tensor)
        nn_hat = nn_hat.reshape(-1).cpu().numpy()
    lgb_hat = lgb_reg.predict(test_feature)
    xgb_hat = xgb_reg.predict(test_feature)
    rf_hat = rf_reg.predict(test_feature)
    # print(type(nn_hat), nn_hat.shape)
    # print(type(lgb_hat), lgb_hat.shape)
    if have_best:
        # return (0.2 * lgb_reg.predict(test_feature) + 0.1 * xgb_reg.predict(test_feature)
        #         + 0.3 * rf_reg.predict(test_feature) + 0.4 * nn_hat)
        return (0.4 * lgb_reg.predict(test_feature) + 0.3 * xgb_reg.predict(test_feature)
                + 0.3 * rf_reg.predict(test_feature))
    else:
        best_mse = 1.0
        best_params = [i for i in range(4)]
        for a in range(1, 8):
            for b in range(1, 9 - a):
                for c in range(1, 10 - a - b):
                    d = 10 - a - b - c
                    ta, tb, tc, td = a / 10, b / 10, c / 10, d / 10
                    pred = ta * lgb_hat + tb * xgb_hat + tc * rf_hat + td * nn_hat
                    pred_mse = mean_squared_error(test_label, pred)
                    if pred_mse < best_mse:
                        best_mse = pred_mse
                        best_params[:] = ta, tb, tc, td
        print(f'The best weights of blendmodels is: {best_params[0]}, {best_params[1]}, '
              f'{best_params[2]}, {best_params[3]}')  # 0.2, 0.1, 0.3, 0.4
        print(f'The best MSE is: {best_mse:.4f}')  # 0.0159
        return (best_params[0] * lgb_hat + best_params[1] * xgb_hat + best_params[2] * rf_hat + best_params[3] * nn_hat)
# label_pred = blend_models_predict(test_feature, test_label)
# MSE_ts = mean_squared_error(test_label, label_pred)
# RMSE_ts = np.sqrt(MSE_ts)
# MAE_ts = mean_absolute_error(test_label, label_pred)
# print("\nBlendModels Pred MSE: {:.4f}\n".format(MSE_ts))  # BlendModels Pred MSE: 0.0093
# print("\nBlendModels Pred RMSE: {:.4f}\n".format(RMSE_ts))  # BlendModels Pred RMSE: 0.0963
# print("\nBlendModels Pred MAE: {:.4f}\n".format(MAE_ts))  # BlendModels Pred MAE: 0.0485

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [clone(x) for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models_)))
        for i, model in enumerate(self.base_models_):
            model.fit(X, y)
            y_pred = model.predict(X)
            out_of_fold_predictions[:, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([model.predict(X) for model in self.base_models_])
        return self.meta_model_.predict(meta_features)

# averaged_models = StackingAveragedModels(base_models=(model_lgb, model_xgb, rf_reg), meta_model=lasso)
# averaged_models.fit(train_feature, train_label)
# label_pred = averaged_models.predict(test_feature)
# MSE_ts = mean_squared_error(test_label, label_pred)
# print("\nAveraged base models MSE: {:.4f}\n".format(MSE_ts))  # Averaged base models MSE: 0.0175

def read_val_image(img_path):
    img_data = rasterio.open(img_path).read()
    band_num, height_num, width_num = np.shape(img_data)

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
    print(os.path.basename(img_path), 'Val读取成功!')

    return feature_data, labels, rows_cols

def predict(model=None, save_path='./result/Blend.xlsx'):
    val_path = './IMGValidation'
    out_path = './IMGPred'

    predictions = []
    val_path_list = os.listdir(val_path)
    for i in range(0, len(val_path_list)):
        img_path = os.path.join(val_path, val_path_list[i])
        # out_name = val_path_list[i].split('.')[0] + "_pred.tif"
        # path = os.path.join(out_path, out_name)
        # print(out_name)

        val_data, val_label, rcs = read_val_image(img_path)
        if model is not None:
            label_hat = model.predict(val_data)
        else:  # model=None默认使用BlendModels预测
            # val_data_tensor = torch.tensor(np.array(val_data), dtype=torch.float32).to(device)
            # label_hat = net(val_data_tensor).reshape(-1).cpu().detach().numpy()
            label_hat = blend_models_predict(test_feature=val_data, have_best=True)
        v, p = pd.Series(val_label), pd.Series(label_hat)
        R_val = v.corr(p)
        MSE_val = mean_squared_error(val_label, label_hat)
        RMSE_val = MSE_val**0.5
        MAE_val = mean_absolute_error(val_label, label_hat)

        # export_pred_img(vl_path=vdp, rows_cols=rcs, val_pred=label_hat, out=out_name)

        res = [val_label.sum(), label_hat.sum(), R_val, MSE_val, RMSE_val, MAE_val]
        predictions.append(res)

    f1 = np.array(predictions)
    vdf = pd.DataFrame(f1)
    vdf.to_excel(save_path, float_format='%.6f', index=0)

# predict(lgb_reg, './result/LGBM.xlsx')
# predict(xgb_reg, './result/XGB.xlsx')
# predict(rf_reg, './result/RF.xlsx')
predict()
