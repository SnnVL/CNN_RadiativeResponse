import numpy as np
import xarray as xr
import utils
import pickle
from data_loader.sample_vault import SampleDict
from utils.DIRECTORIES import DATA_DIRECTORY, SHAPE_DIRECTORY, MODEL_DIRECTORY
from scipy.signal import butter, filtfilt, detrend

__author__ = "Senne Van Loon"
__version__ = "24 April 2026"

# https://github.com/eabarnes1010/shash_peak_warming_public/blob/main/datamaker/data_generator.py

class ClimateData:
    """
    Custom dataset for climate data and processing.
    """

    def __init__(
        self, config, expname, seed, fetch=True, verbose=False
    ):

        self.config = config
        self.expname = expname
        self.seed = seed
        self.verbose = verbose
        self.input_var = config["input_var"]
        self.label_var = config["label_var"]

        if self.config.get("fixed_seed", None) is not None:
            self.rng = np.random.default_rng(self.config.get("fixed_seed") + 42)
        else:
            self.rng = np.random.default_rng(self.seed + 42)

        if fetch:
            self.fetch_data()


    def fetch_data(self, verbose=None):
        if verbose is not None:
            self.verbose = verbose

        self.d_train = SampleDict()
        self.d_val = SampleDict()
        self.d_test = SampleDict()

        self._get_members()
        self._create_data()

        if self.verbose:
            print("*** Training Data Summary ***")
            self.d_train.summary()
            print("*** Validation Data Summary ***")
            self.d_val.summary()
            print("*** Testing Data Summary ***")
            self.d_test.summary()


    def _create_data(self):

        # Loop over all models
        for im, model in enumerate(self.config["models"]):
            if self.verbose:
                print("Loading data from "+model)

            # Load DataArrays
            da_input = utils.load_data(\
                model+'/'+self.input_var, \
                self.config["data_periods"][im],
                self.config \
            )
            da_label = utils.load_data(\
                model+'/'+self.label_var, \
                self.config["data_periods"][im],
                self.config, \
            )

            if "frequency_filter" in self.config.keys():
                da_input = self._filter_data(da_input)
                da_label = self._filter_data(da_label)
            
            # Anomalies
            if "anomaly_list" in self.config.keys():
                anom = self.config["anomaly_list"][im]
            else:
                anom = self.config["anomalies"]
            da_input = self._get_anomalies(da_input, anom, var=model+'/'+self.input_var)
            da_label = self._get_anomalies(da_label, anom, var=model+'/'+self.label_var)

            # Grab years to train
            da_input = da_input.sel(
                time=slice(self.config["date_bounds"][0],self.config["date_bounds"][-1])
            ).fillna(0.)
            da_label = da_label.sel(
                time=slice(self.config["date_bounds"][0],self.config["date_bounds"][-1])
            ).fillna(0.)

            # Masking
            if self.config["input_mask"]:
                da_input = self._mask_data(da_input, self.config["input_mask"])
            if self.config["label_mask"]:
                da_label = self._mask_data(da_label, self.config["label_mask"])

            # Add channel dimension
            # Add this point, dim of da is [member, time, lat, lon]
            # This changes the dimensions to [member, time, channel, lat, lon]
            da_input = da_input.expand_dims(dim={"channel": 1}, axis=2)
            if self.config["map_output"]:
                da_label = da_label.expand_dims(dim={"channel": 1}, axis=2)
            
            # Split data
            if self.config["split_by_years"]:

                # Assert that we only split by years when n_train=1, n_val=0, n_test=0
                assert self.config["n_train_val_test"][0] == 1, "Splitting by years only when n_train=1"
                assert self.config["n_train_val_test"][1] == 0, "Splitting by years only when n_val=0"
                assert self.config["n_train_val_test"][2] == 0, "Splitting by years only when n_test=0"

                if isinstance(self.config['split_periods']['train'], int):
                    self._get_random_years()
                    f_dict_train = self._split_by_years(da_input, da_label, self.train_years, model)
                    f_dict_val = self._split_by_years(da_input, da_label, self.val_years, model)
                    f_dict_test = self._split_by_years(da_input, da_label, self.test_years, model)
                else:
                    f_dict_train = self._split_by_years(da_input, da_label, self.config['split_periods']['train'], model)
                    f_dict_val = self._split_by_years(da_input, da_label, self.config['split_periods']['val'], model)
                    f_dict_test = self._split_by_years(da_input, da_label, self.config['split_periods']['test'], model)
            else:
                f_dict_train = self._get_dict_data(da_input, da_label, self.train_members, model)
                f_dict_val = self._get_dict_data(da_input, da_label, self.val_members, model)
                f_dict_test = self._get_dict_data(da_input, da_label, self.test_members, model)

            # # concatenate with the rest of the data
            self.d_train.concat(f_dict_train)
            self.d_val.concat(f_dict_val)
            self.d_test.concat(f_dict_test)

        # reshape the data into samples
        self.d_train.reshape()
        self.d_val.reshape()
        self.d_test.reshape()

        # Get data around zero
        if "subtract_val" in self.config.keys():
            model_name = utils.get_model_name(self.expname, self.seed)
            if self.config["subtract_val"] == 'save':
                print("** Saving training mean.")
                x_subtract = self.d_train["x"].mean(axis=0)
                y_subtract = self.d_train["y"].mean(axis=0)
                with open(MODEL_DIRECTORY+model_name+".pickle", 'wb') as f:
                    pickle.dump(x_subtract, f)
                    pickle.dump(y_subtract, f)
            elif self.config["subtract_val"] == 'load':
                print("** Loading training mean.")
                with open(MODEL_DIRECTORY+model_name+".pickle", 'rb') as f:
                    x_subtract = pickle.load(f)
                    y_subtract = pickle.load(f)
            elif self.config["subtract_val"][-7:] == ".pickle":
                print("** Loading training mean.")
                with open(MODEL_DIRECTORY+self.config["subtract_val"], 'rb') as f:
                    x_subtract = pickle.load(f)
                    y_subtract = pickle.load(f)
            else:
                raise RuntimeError("Pickle value necessary for subtract_val.")
            self.d_train["x"] += -x_subtract[np.newaxis,...]
            self.d_val["x"]   += -x_subtract[np.newaxis,...]
            self.d_test["x"]  += -x_subtract[np.newaxis,...]
            self.d_train["y"] += -y_subtract[np.newaxis,...]
            self.d_val["y"]   += -y_subtract[np.newaxis,...]
            self.d_test["y"]  += -y_subtract[np.newaxis,...]

        # add latitude and longitude
        self.input_lat = da_input.lat.values
        self.input_lon = da_input.lon.values

        if self.config["map_output"]:
            self.label_lat = da_label.lat.values
            self.label_lon = da_label.lon.values


    def _get_anomalies(self, da, anom, var=None):

        # Anomalies
        if anom == 'years':
            da_mean = da.sel(
                time=slice(self.config["anomaly_dates"][0], self.config["anomaly_dates"][-1])
            ).mean(dim=('time', 'member'))

        elif anom == 'years_member':

            da_mean = da.sel(
                time=slice(self.config["anomaly_dates"][0], self.config["anomaly_dates"][-1])
            ).mean(dim=('time'))

        elif anom is False:
            
            da_mean = xr.zeros_like(da)
            
        elif anom[-3:] == ".nc":

            da_mean = xr.open_dataarray(DATA_DIRECTORY + self.config["datafolder"] + var + anom)
            
        else:

            da_mean = xr.zeros_like(da)

        return (da - da_mean)
    

    def _mask_data(self, da, mask_file):

        msk = xr.open_dataarray(SHAPE_DIRECTORY + mask_file)

        return da*msk
    
    
    def _get_dict_data(self, da_input, da_label, members, model):

        f_dict = SampleDict()

        f_dict["x"] = da_input.sel(member=members).values       
        f_dict["y"] = da_label.sel(member=members).values 

        f_dict["year"] = np.tile(
            da_input["time.year"].values, 
            (f_dict["y"].shape[0], 1)
        )
        f_dict["member"] = np.tile(
            members[:, np.newaxis], 
            (1,f_dict["y"].shape[1])
        )
        f_dict["model"] = np.tile(
            model, 
            (f_dict["y"].shape[0], f_dict["y"].shape[1])
        )

        return f_dict
    
    def _split_by_years(self, da_input, da_label, periods, model):

        f_dict = SampleDict()

        if isinstance(periods[0], np.int64):
            da_x = da_input.isel(member=[0,],time=periods)
            da_y = da_label.isel(member=[0,],time=periods)
        else:
            da_x = []
            da_y = []
            for period in periods:
                assert len(period) == 2, "Each period must have a start and end time: "+str(period)

                da_x.append(
                    da_input.sel(member=[0,],time=slice(period[0],period[1]))
                )
                da_y.append(
                    da_label.sel(member=[0,],time=slice(period[0],period[1]))
                )
            da_x = xr.concat(da_x, dim='time')
            da_y = xr.concat(da_y, dim='time')
        
        f_dict["x"] = da_x.values       
        f_dict["y"] = da_y.values 

        f_dict["year"] = np.tile(
            da_x["time.year"].values, 
            (f_dict["y"].shape[0], 1)
        )
        f_dict["member"] = np.tile(
            0, 
            (f_dict["y"].shape[0], f_dict["y"].shape[1])
        )
        f_dict["model"] = np.tile(
            model, 
            (f_dict["y"].shape[0], f_dict["y"].shape[1])
        )

        return f_dict


    def _get_random_years(self):

        # get number of members or fraction of members
        n_train = self.config["split_periods"]["train"]
        n_val = self.config["split_periods"]["val"]
        n_test = self.config["split_periods"]["test"]

        all_years = np.arange(0, n_train + n_val + n_test)

        # Random number generator
        rng_cmip = np.random.default_rng(self.seed)

        # Select random members
        self.train_years = np.sort(rng_cmip.choice(all_years, size=n_train, replace=False))
        self.val_years = np.sort(rng_cmip.choice(np.setdiff1d(all_years, self.train_years), size=n_val, replace=False))
        self.test_years = np.sort(rng_cmip.choice(np.setdiff1d(all_years, np.append(self.train_years[:], self.val_years)), size=n_test, replace=False))

        if self.verbose:
            print(
                f"Years for train/val/test split: {self.train_years} / {self.val_years} / {self.test_years}"
            )

    def _get_members(self):

        # get number of members or fraction of members
        n_train = self.config["n_train_val_test"][0]
        n_val = self.config["n_train_val_test"][1]
        n_test = self.config["n_train_val_test"][2]

        all_members = np.arange(0, n_train + n_val + n_test)

        # Random number generator
        rng_cmip = np.random.default_rng(self.seed)

        # Select random members
        self.train_members = np.sort(rng_cmip.choice(all_members, size=n_train, replace=False))
        self.val_members = np.sort(rng_cmip.choice(np.setdiff1d(all_members, self.train_members), size=n_val, replace=False))
        self.test_members = np.sort(rng_cmip.choice(np.setdiff1d(all_members, np.append(self.train_members[:], self.val_members)), size=n_test, replace=False))

        if self.verbose:
            print(
                f"Member for train/val/test split: {self.train_members} / {self.val_members} / {self.test_members}"
            )

    def _filter_data(self, da):

        if self.config["frequency_filter"]["cutoff_period"] == 0.0:
            # Linearly detrend
            filtered_data = detrend(da.copy(), axis=da.get_axis_num('time'))
        else:
            # Butter filter
            b, a = butter(
                self.config["frequency_filter"]["order"], 
                2 / self.config["frequency_filter"]["cutoff_period"], 
                btype=self.config["frequency_filter"]["type"], 
                analog=False
            )
            filtered_data = filtfilt(b, a, da.copy(), axis=da.get_axis_num('time'))
        
        da[:] = filtered_data

        return da
    


class ObsData:
    """
    Custom dataset for observational data with inputs only (no labels).
    """

    def __init__(self, config, expname, seed, fetch=True, verbose=False):
        self.config = config
        self.expname = expname
        self.seed = seed
        self.verbose = verbose
        self.input_var = config["input_var"]

        if self.config.get("fixed_seed", None) is not None:
            self.rng = np.random.default_rng(self.config.get("fixed_seed") + 42)
        else:
            self.rng = np.random.default_rng(self.seed + 42)

        if fetch:
            self.fetch_data()

    def fetch_data(self, verbose=None):
        if verbose is not None:
            self.verbose = verbose

        self._create_data()

        if self.verbose:
            print("*** Observational Data Summary ***")
            self.d_obs.summary()

    def _create_data(self):
        
        # Load DataArray
        da_input = utils.load_data(\
            self.config['data_source']+'/'+self.input_var, \
            self.config["data_period"],
            self.config \
        )

        # Filter
        if "frequency_filter" in self.config.keys():
            da_input = self._filter_data(da_input)

        # Anomalies
        da_input = self._get_anomalies(da_input, self.config["anomalies"])

        # Grab years to train
        da_input = da_input.sel(
            time=slice(self.config["date_bounds"][0],self.config["date_bounds"][-1])
        ).fillna(0.)

        # Masking
        if self.config["input_mask"]:
            da_input = self._mask_data(da_input, self.config["input_mask"])

        # Add channel dimension
        if self.config['multi_mem']:
            # Add this point, dim of da is [member, time, lat, lon]
            # This changes the dimensions to [member, time, channel, lat, lon]
            da_input = da_input.expand_dims(dim={"channel": 1}, axis=2)
        else:
            # Add this point, dim of da is [time, lat, lon]
            # This changes the dimensions to [time, channel, lat, lon]
            da_input = da_input.expand_dims(dim={"channel": 1}, axis=1)
            
        self.d_obs = self._get_dict_data(da_input)

        if self.config['multi_mem']:
            self.d_obs.reshape()

        # add latitude and longitude
        self.input_lat = da_input.lat.values
        self.input_lon = da_input.lon.values


    def _get_anomalies(self, da, anom, var=None):

        # Anomalies
        if anom == 'years':
            da_mean = da.sel(
                time=slice(self.config["anomaly_dates"][0], self.config["anomaly_dates"][-1])
            ).mean(dim=('time'))

        elif anom is False:
            
            da_mean = xr.zeros_like(da)
            
        elif anom[-3:] == ".nc":

            da_mean = xr.open_dataarray(DATA_DIRECTORY + self.config["datafolder"] + var + anom)
            
        else:

            da_mean = xr.zeros_like(da)

        return (da - da_mean)
    
    def _mask_data(self, da, mask_file):

        msk = xr.open_dataarray(SHAPE_DIRECTORY + mask_file)

        return da*msk
    
    def _get_dict_data(self, da_input):

        f_dict = SampleDict()


        # Get data around zero
        if "subtract_val" in self.config.keys():
            if self.config["subtract_val"][-7:] == ".pickle":
                print("** Loading training mean.")
                with open(MODEL_DIRECTORY+self.config["subtract_val"], 'rb') as f:
                    x_subtract = pickle.load(f)
            else:
                raise RuntimeError("Pickle value necessary for subtract_val.")
            f_dict["x"] = da_input.values - x_subtract[np.newaxis,...]
        else:
            f_dict["x"] = da_input.values

        f_dict["y"] = np.full(f_dict["x"].shape[:2],np.nan)

        if self.config['multi_mem']:
            f_dict["year"] = np.tile(
                da_input["time.year"].values, 
                (f_dict["y"].shape[0], 1)
            )
        else:
            f_dict["year"] = np.tile(
                da_input["time.year"].values, 
                (1, 1)
            )
            
        return f_dict
    
    def _filter_data(self, da):

        if self.config["frequency_filter"]["cutoff_period"] == 0.0:
            # Linearly detrend
            filtered_data = detrend(da.copy(), axis=da.get_axis_num('time'))
        else:
            # Butter filter
            b, a = butter(
                self.config["frequency_filter"]["order"], 
                2 / self.config["frequency_filter"]["cutoff_period"], 
                btype=self.config["frequency_filter"]["type"], 
                analog=False
            )
            filtered_data = filtfilt(b, a, da.copy(), axis=da.get_axis_num('time'))
        
        da[:] = filtered_data

        return da