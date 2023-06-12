from pylab import *
import xarray as xr
import rioxarray
import torch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import cmaps
import os
import rasterio
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

path = './IMGTrain/130036_20150111.tif'

val_path = 'IMGValidation'
pred_path = 'IMGPred'
val_all_img_path = os.listdir(val_path)
pred_all_img_path = os.listdir(pred_path)
dates = []
dates_to_names = {}
for i in range(0, len(val_all_img_path)):
    date = val_all_img_path[i].split('.')[0].split('_')[1]
    dates.append(date)
    dates_to_names[date] = val_all_img_path[i]
dates.sort()
dates_to_names = dict(sorted(dates_to_names.items(), key=lambda x: x[0], reverse=False))

dem = rioxarray.open_rasterio(path)
# print(dem.shape)

def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)

# dem = dem[0]
# dem_norm = normalize(dem)
# plt.imshow(dem, cmap='terrain')
# plt.show()

# plt.figure(dpi=300, figsize=(12, 4))
# plt.subplots_adjust(hspace=0.2, wspace=0.5)
# plt.subplot(1, 2, 1)
# dem[20].plot(cmap='terrain')  # getting the first band
# plt.subplot(1, 2, 2)
# dem[21].plot(cmap='terrain')
# # plt.savefig('1.png',dpi=800,bbox_inches='tight',pad_inches=0)
# plt.show()

# dem = torch.tensor(np.array(dem.values), dtype=torch.float32)
# print(dem.shape)

# dem = torch.permute(dem, (1, 2, 0))
# print(dem.shape)
# for i in range(dem.shape[0]):
#     for j in range(dem.shape[1]):
#         print(dem[i][j])

# dem = torch.where(torch.isnan(dem), torch.full_like(dem, 0), dem)
# print(dem[0][0])

img_data = rasterio.open(path).read()
# print(np.shape(img_data[0]))
img_shape = np.shape(img_data)
print(img_shape, type(img_shape))  # (22, 482, 477) <class 'tuple'>
img_shape = f'({img_shape[0]}, {img_shape[1]}, {img_shape[2]})'
print(img_shape, type(img_shape))  # (22, 482, 477) <class 'str'>
