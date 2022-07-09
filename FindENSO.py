import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
eqsst = xr.open_dataset("./data/eqsst.nc")["ObsSST"]
# 计算Nino3.4指数
Nino34 = eqsst.loc[:, 190:240].mean("lon")
# 计算三月滑动平均
eqsst = eqsst.rolling(time=3, center=True).mean()
# 计算ONDJF的Nino3.4指数
Nino34ONIDJF = Nino34.rolling(time=5, center=True).mean().groupby("time.month")[12][:-1]
# 获得厄尔尼诺年
year = Nino34ONIDJF[Nino34ONIDJF >= 0.6].time.dt.year
print(year)
# 获得厄尔尼诺年用于计算和绘图的数据
draw_ls = []
cal_ls = []
for yr in np.array(year):
    time1 = "%s-10-01" % (yr - 1)
    time2 = "%s-10-30" % yr
    time22 = "%s-03-30" % (yr + 1)
    # 选取需要的数据
    dat_ssta = eqsst.loc[time1:time2]
    dr_ssta = eqsst.loc[time1:time22]
    draw_ls.append(np.array(dr_ssta)[np.newaxis, :])
    cal_ls.append(np.array(dat_ssta)[np.newaxis, :])
# 合并
draw_ls = np.concatenate(draw_ls, axis=0)
cal_ls = np.concatenate(cal_ls, axis=0)
print(draw_ls.shape, cal_ls.shape)
# 保存为nc文件
draw_xr = xr.DataArray(draw_ls, coords={"year": np.array(year), "month": np.arange(18), "lon": eqsst.lon})
cal_xr = xr.DataArray(cal_ls, coords={"year": np.array(year), "month": np.arange(13), "lon": eqsst.lon})
draw_ds = xr.Dataset({"ad_d": draw_xr})
cal_ds = xr.Dataset({"ad": cal_xr})
cal_ds.to_netcdf("./data/ad.nc")
draw_ds.to_netcdf("./data/ad_d.nc")
