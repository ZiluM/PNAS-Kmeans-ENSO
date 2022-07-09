import xarray as xr

data_path = "../DATA/"

# 选定所需的范围
hs = xr.open_dataset(data_path + "HadISST_sst_2x2.nc")['sst'].loc \
                ["1900":"2018", -5:5, 120:280]
es = xr.open_dataset(data_path + "ersstv5_1870-2019_2x2.nc")['sst'].loc \
                ["1900":"2018", 0, -5:5, 120:280]
hs = hs.mean("lat")
es = es.mean("lat")
es = es.interpolate_na(dim="lon", method="linear", fill_value="extrapolate")
hs = hs.interpolate_na(dim="lon", method="linear", fill_value="extrapolate")

hs['time'] = es['time']

hs_p_es = (hs + es) / 2.
hs_p_es_a = hs_p_es.groupby("time.month") - hs_p_es.groupby("time.month").mean()
# hs_p_es_a = hs_p_es_a.rolling(time=3,center = True).mean()
# hs_p_es_a = hs_p_es_a.loc["1900":"2017"]

eqsst = xr.Dataset({"ObsSST":hs_p_es_a})
eqsst.to_netcdf("./data/eqsst.nc")
#
