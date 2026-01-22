import numpy as np
import xarray as xr
import yaml
import sys
import os
import re

def GetFilenamesFromYaml(yml, extract_lam=False):
    '''
        Function for obtaining various information from inference yaml file. 
        Finds path and filename of original netCDF inference file. 
        Creates a new filename for netCDF file from information in yaml file, following this convention:
        YYYY-MM-DD-<last 5 run ID symbols>-e<epoch number>-s<step number>.nc
    Args:
        yml             [str]   :   Yaml file used for the inference
        exctract_lam    [bool]  :   Set to True if lam region is extracted in yaml file.
    Returns:
        info_dict       [dict]  :   A dictionary containing the original netCDF filename, it's path and new filename based on conventions. 

    '''
    with open(yml, 'r') as file:
        
        data = yaml.safe_load(file)

        if extract_lam is False:
            orig_file = data['output']['netcdf']
        elif extract_lam is True:
            orig_file = data['output']['extract_lam']['netcdf']['path']

        run_id = re.findall(r'(?<=checkpoint\/).*(?=\/)', data['checkpoint'])[0][-5:]
        epoch = re.findall(r'(?<=epoch\_).*(?=\-)', data['checkpoint'])[0]
        step = re.findall(r'(?<=step\_).*(?=\.)', data['checkpoint'])[0]
        lead_time = data['lead_time'] 

        orig_file_path = re.findall(r'.*(?<=\/)', orig_file)[0]
        convention_filename = str(data['date'])+'_'+lead_time+'_'+run_id+'_e'+epoch+'_s'+step+'.nc'

        info_dict = {
            'orig_file':orig_file,
            'orig_file_path':orig_file_path,
            'convention_filename':convention_filename
        }
        
    return info_dict

def InferenceTo2D(file, output=None, var_list=None, clean1D=True, grid_file = '/pfs/lustrep3/scratch/project_465002266/datasets/norkyst_grd_v31.nc'):
    '''
        Function for reshaping 1D inference fields to 2D. Currently uses Norkyst grid file for this, meaning that the region size must be the safe as original Norkyst grid. 
        extract_lam in inference currently doesn't work with this. 
    Args:
        file        [str]   :   Input 1D inference .nc file. 
        output      [str]   :   Name of output .nc file. If None uses input filename (overwrites).
        var_list    [list]  :   List of variables to include in output .nc file. If None includes all found variables. 
        clean1D     [bool]  :   If True deletes original file. 
        grid_file   [str]   :   Norkyst grid file. 
    '''
    
    ds = xr.open_dataset(file)
    grid = xr.open_dataset(grid_file)
    time = ds["time"].values

    if np.issubdtype(time.dtype, np.datetime64):
        time_numeric = (time - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "D")
    else:
        time_numeric = time
    new_vars = {}

    if var_list is None:
            var_list = ds.data_vars

    shape = [len(grid.eta_rho.values),len(grid.xi_rho.values)]
    for var in ds.data_vars:
        data = ds[var].values
        if var in var_list:
            if len(np.shape(ds[var])) == 1:
                reshaped = data.reshape(shape[0], shape[1])
                new_vars[var]= xr.DataArray(
                    reshaped,
                    coords={'X': grid.eta_rho.values, 'Y':grid.xi_rho.values},
                    name=var
                )
            elif len(np.shape(ds[var])) == 2:
                reshaped = data.reshape((len(time), shape[0], shape[1]))
                new_vars[var] = xr.DataArray(
                    reshaped,
                    coords={'time': time_numeric, "X": grid.eta_rho.values, "Y": grid.xi_rho.values},
                    name=var
                )

    new_ds = xr.Dataset(new_vars)
    new_ds["time"].attrs["units"] = "days since 1970-01-01 00:00:00"
    new_ds["time"].attrs["calendar"] = "standard"

    if clean1D is True:
        os.remove(file)

    if output is None:
        output = file
    new_ds.to_netcdf(output)

if __name__ == '__main__':
    print('Now reshaping inference file.')
    info_dict = GetFilenamesFromYaml(sys.argv[1])
    InferenceTo2D(info_dict['orig_file'], info_dict['orig_file_path']+'/'+str(info_dict['convention_filename']), grid_file='/home/havis/sea/ROMS/metroms_apps/norkyst_v3/norkyst_oper/Run/Grid/norkyst_grd_v31.nc')
