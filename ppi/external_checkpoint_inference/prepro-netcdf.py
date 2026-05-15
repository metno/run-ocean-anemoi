

def get_daily_ocean_files():
	"""Get a list of daily NetCDF files from the data directory."""
	import datetime
	start = datetime.datetime.now()
	path = f'/lustre/storeB/project/fou/hi/oper/norkyst_v3/forecast/his_zdepths/{start.year}/{start.month:02d}/{start.day:02d}/'
	file_list = [
        path+f'norkyst800_his_zdepth_{start.year}{start.month:02d}{start.day:02d}T00Z_m00_AN.nc',
        path+f'norkyst800_his_zdepth_{start.year}{start.month:02d}{start.day:02d}T00Z_m00_FC_0001.nc',
        #path+f'norkyst800_his_sdepth_{start.year}{start.month:02d}{start.day:02d}T00Z_m00_FC_0002.nc',
        #path+f'norkyst800_his_sdepth_{start.year}{start.month:02d}{start.day:02d}T00Z_m00_FC_0003.nc',
        #path+f'norkyst800_his_sdepth_{start.year}{start.month:02d}{start.day:02d}T00Z_m00_FC_0004.nc'
    ]
	return file_list

def get_daily_atmos_files():
    """Get a list of daily NetCDF files from the data directory."""


def concat_files(file_list, depth=None):
    """Concatenate a list of NetCDF files into a single xarray Dataset."""
    import xarray as xr
    ds = xr.open_mfdataset(file_list, combine='by_coords')
    if depth is not None:
        try:
            ds = ds.isel(depth=depth)
        except:
            ds = ds.isel(s_rho=depth, s_w=depth)
    return ds

def flatten_dataarray(ds, dims=['X', 'Y']):
    """Flatten a Dataset with multiple dimensions into a 2D DataArray."""
    import xarray as xr
    ds = ds.stack(values=dims)
    return ds

def rename_variables(ds):
    """Rename variables in the Dataset to match the expected format."""
    depth = int(ds['depth'].values)
    ds = ds.rename({'lon': 'longitude', 
                    'lat': 'latitude',
                    'temperature': f'temperature_{depth}',
                    'salinity': f'salinity_{depth}',
                    'u_eastward': f'u_eastward_{depth}',
                    'v_northward': f'v_northward_{depth}'})
    return ds

if __name__ == "__main__":
    ds = concat_files(get_daily_ocean_files(), depth=0)
    ds = flatten_dataarray(ds)
    ds = rename_variables(ds)
    print(ds)