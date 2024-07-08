import torch
from torch.utils.data import Dataset
import os
import numpy as np

LABELS_NAMES = ['barley', 'wheat', 'rapeseed', 'corn', 'sunflower', 'orchards',
       'nuts', 'perm. mead', 'temp. mead']

APPROXIMATED_DOY_51 = [2, 12, 15, 32, 42, 52, 62, 72, 82, 92, 102, 112, 122, 132, 142, 152, 162, 172, \
                       182, 187, 192, 197, 202, 207, 212, 217, 227, 232, 237, 242, 252, 257, 262, 267, 272, 277, \
                       282, 292, 297, 302, 307, 312, 317, 322, 327, 332, 342, 347, 352, 357, 362] 

APPROXIMATED_DOY_102 = [2, 5, 12, 15, 25, 32, 35, 42, 45, 52, 55, 62, 65, 72, 75, 82, 85, 92, \
                        95, 102, 105, 112, 115, 122, 125, 132, 135, 142, 145, 152, 155, 162, 165, 172, 175, 180, \
                        182, 185, 187, 190, 192, 195, 197, 200, 202, 205, 207, 210, 212, 215, 217, 220, 225, 227, \
                        230, 232, 235, 237, 242, 245, 250, 252, 255, 257, 260, 262, 265, 267, 270, 272, 275, 277, \
                        280, 282, 292, 295, 297, 300, 302, 305, 307, 310, 312, 315, 317, 320, 322, 325, 327, 330, \
                        332, 335, 340, 342, 345, 347, 350, 352, 355, 357, 360, 362]

class BreizhCrops(Dataset):
    def __init__(self, partition="train", root="breizhcrops_dataset", sequencelength=70, year=2017, return_id=False, corrected=False, daily_timestamps=False, original_time_serie_lengths=[51, 102], preload_ram=True):
        """
        BreizhCrops dataset 
        INPUT: 
        - partition: str, either "train", "valid" or "eval"
        - root: str, path to the root folder where the data is stored
        - sequencelength: int, the length of the sequences that will be returned
        - year: int, the year of the data, either 2017 or 2018
        - return_id: bool, if True, the id of the field will be returned with the data
        - corrected: bool, if True, only time series with lengths "original_time_serie_lengths" will be kept
        - daily_timestamps: bool, if True, the sequences will be of length 365, corresponding to the days of the year. The time series will be padded with zeros
        """
        assert partition in ["train", "valid", "eval", "final_train"]
        if not corrected:
            if partition == "train":
                frh01 = BzhBreizhCrops("frh01", root=root, transform=lambda x: x, preload_ram=preload_ram, year=year, corrected=corrected)
                frh02 = BzhBreizhCrops("frh02", root=root, transform=lambda x: x, preload_ram=preload_ram, year=year, corrected=corrected)
                self.ds = torch.utils.data.ConcatDataset([frh01, frh02])
                self.labels_names = frh01.labels_names
            elif partition == "valid":
                self.ds = BzhBreizhCrops("frh03", root=root, transform=lambda x: x, preload_ram=preload_ram, year=year, corrected=corrected)
                self.labels_names = self.ds.labels_names
            elif partition == "eval":
                self.ds = BzhBreizhCrops("frh04", root=root, transform=lambda x: x, preload_ram=preload_ram, year=year, corrected=corrected)
                self.labels_names = self.ds.labels_names
            elif partition == "final_train":
                # group validation and train set together
                frh01 = BzhBreizhCrops("frh01", root=root, transform=lambda x: x, preload_ram=preload_ram, year=year, corrected=corrected)
                frh02 = BzhBreizhCrops("frh02", root=root, transform=lambda x: x, preload_ram=preload_ram, year=year, corrected=corrected)
                frh03 = BzhBreizhCrops("frh03", root=root, transform=lambda x: x, preload_ram=preload_ram, year=year, corrected=corrected)
                self.ds = torch.utils.data.ConcatDataset([frh01, frh02, frh03])
                self.labels_names = frh01.labels_names
        else:
            # because of the corrected flag, we need to load the datasets differently for the sizes to be reasonable
            if partition == "train":
                self.ds = BzhBreizhCrops("frh02", root=root, transform=lambda x: x, preload_ram=preload_ram, year=year, corrected=corrected, original_time_serie_lengths=original_time_serie_lengths)
                self.labels_names = self.ds.labels_names
            elif partition == "valid":
                self.ds = BzhBreizhCrops("frh01", root=root, transform=lambda x: x, preload_ram=preload_ram, year=year, corrected=corrected, original_time_serie_lengths=original_time_serie_lengths)
                self.labels_names = self.ds.labels_names
            elif partition == "eval":
                frh03 = BzhBreizhCrops("frh03", root=root, transform=lambda x: x, preload_ram=preload_ram, year=year, corrected=corrected, original_time_serie_lengths=original_time_serie_lengths)
                frh04 = BzhBreizhCrops("frh04", root=root, transform=lambda x: x, preload_ram=preload_ram, year=year, corrected=corrected, original_time_serie_lengths=original_time_serie_lengths)
                self.ds = torch.utils.data.ConcatDataset([frh03, frh04])
                self.ds_1 = frh03
                self.ds_2 = frh04
                self.labels_names = frh03.labels_names
            elif partition == "final_train":
                # group validation and train set together
                frh01 = BzhBreizhCrops("frh01", root=root, transform=lambda x: x, preload_ram=preload_ram, year=year, corrected=corrected, original_time_serie_lengths=original_time_serie_lengths)
                frh02 = BzhBreizhCrops("frh02", root=root, transform=lambda x: x, preload_ram=preload_ram, year=year, corrected=corrected, original_time_serie_lengths=original_time_serie_lengths)
                self.ds = torch.utils.data.ConcatDataset([frh01, frh02])
                self.labels_names = frh01.labels_names

        self.corrected = corrected
        self.daily_timestamps = daily_timestamps 
        if self.daily_timestamps: 
            self.sequencelength = 365
        else: 
            self.sequencelength = sequencelength
        self.return_id = return_id
        self.class_weights = None
        self.nclasses = len(self.labels_names)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        X,y,id = self.ds[item]

        # take bands and normalize
        # ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12', '...']
        X = X[:,:13] * 1e-4

        # get length of this sample
        t = X.shape[0]

        if self.corrected and self.daily_timestamps:
            # in this case the sequences are either of length 51 or 102.
            if t==51:
                doys = APPROXIMATED_DOY_51
            elif t==102:
                doys = APPROXIMATED_DOY_102
            else:
                raise ValueError(f"Sequence length {t} not recognized")
            # change the length of X to be 365, and pad with zeros
            X_ = np.zeros((365, X.shape[1]))
            X_[doys] = X
            X = X_
            t = 365
        else: 
            if t < self.sequencelength:
                # time series shorter than "sequencelength" will be zero-padded
                npad = self.sequencelength - t
                X = np.pad(X, [(0, npad), (0, 0)], 'constant', constant_values=0)
            elif t > self.sequencelength:
                # time series longer than "sequencelength" will be sub-sampled
                idxs = np.random.choice(t, self.sequencelength, replace=False)
                idxs.sort()
                X = X[idxs]

        X = torch.from_numpy(X).type(torch.FloatTensor)

        X, y = X, y.repeat(self.sequencelength)
        if self.return_id:
            return X, y, id
        else:
            return X, y
        
    def get_sequence_lengths(self):
        """ 
        Returns the sequence lengths of the time series in the dataset
        """
        if hasattr(self.ds, "datasets"):
            sequence_length = []
            for ds in self.ds.datasets:
                sequence_length.append(np.array(ds.index['sequencelength'].values))
            sequence_length = np.concatenate(sequence_length)
        else:
            sequence_length = np.array(self.ds.index['sequencelength'].values)
        return np.array(sequence_length)
    
    def get_class_weights(self):
        """
        Returns the class weights of the dataset
        The class with the smallest count will have weight 1, others will have a weight < 1
        """
        if self.class_weights is None:
            class_counts = torch.zeros(self.nclasses)
            for i in range(len(self.ds)):
                _, y, _ = self.ds[i]
                class_counts[y] += 1
            min_class_count = class_counts.min()
            class_weights = min_class_count / class_counts 
            self.class_weights = class_weights
        return self.class_weights


import os
import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import numpy as np


BANDS = {
    "L1C": ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
            'B8A', 'B9', 'QA10', 'QA20', 'QA60', 'doa', 'label', 'id'],
    "L2A": ['doa', 'id', 'code_cultu', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
            'B8A', 'B11', 'B12', 'CLD', 'EDG', 'SAT']
}

SELECTED_BANDS = {
    "L1C": ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12',
            'QA10', 'QA20', 'QA60', 'doa'],
    "L2A": ['doa', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12',
            'CLD', 'EDG', 'SAT', ]
}

PADDING_VALUE = -1


class BzhBreizhCrops(Dataset):

    def __init__(self,
                 region,
                 root="breizhcrops_dataset",
                 year=2017, level="L1C",
                 transform=None,
                 target_transform=None,
                 filter_length=0,
                 verbose=False,
                 load_timeseries=True,
                 recompile_h5_from_csv=False,
                 preload_ram=False,
                 corrected=False,
                 original_time_serie_lengths=[51, 102],
                 ):
        """
        :param region: dataset region. choose from "frh01", "frh02", "frh03", "frh04", "belle-ile"
        :param root: where the data will be stored. defaults to `./breizhcrops_dataset`
        :param year: year of the data. currently only `2017`
        :param level: Sentinel 2 processing level. Either `L1C` (top of atmosphere) or `L2A` (bottom of atmosphere)
        :param transform: a transformation function applied to the raw data before retrieving a sample. Can be used for featured extraction or data augmentaiton
        :param target_transform: a transformation function applied to the label.
        :param filter_length: time series shorter than `filter_length` will be ignored
        :param bool verbose: verbosity flag
        :param bool load_timeseries: if False, no time series data will be loaded. Only index file and class initialization. Used mostly for tests
        :param bool recompile_h5_from_csv: downloads raw csv files and recompiles the h5 databases. Only required when dealing with new datasets
        :param bool preload_ram: loads all time series data in RAM at initialization. Can speed up training if data is stored on HDD.
        :param bool corrected: if True, only time series with 51 or 102 length will be kept
        """
        assert year in [2017, 2018]
        assert level in ["L1C", "L2A"]
        assert region in ["frh01", "frh02", "frh03", "frh04", "belle-ile"]

        if transform is None:
            transform = get_default_transform(level)
        if target_transform is None:
            target_transform = get_default_target_transform()
        self.transform = transform
        self.target_transform = target_transform

        self.region = region.lower()
        self.bands = BANDS[level]

        self.verbose = verbose
        self.year = year
        self.level = level
        self.corrected = corrected

        if verbose:
            print(f"Initializing BreizhCrops region {region}, year {year}, level {level}")

        self.root = root
        self.h5path, self.indexfile, self.codesfile, self.shapefile, self.classmapping, self.csvfolder = \
            self.build_folder_structure(self.root, self.year, self.level, self.region)

        self.load_classmapping(self.classmapping)

        if os.path.exists(self.h5path):
            print(os.path.getsize(self.h5path), FILESIZES[year][level][region])
            h5_database_ok = os.path.getsize(self.h5path) == FILESIZES[year][level][region]
        else:
            h5_database_ok = False

        if not os.path.exists(self.indexfile):
            download_file(INDEX_FILE_URLs[year][level][region], self.indexfile)

        if not h5_database_ok and recompile_h5_from_csv and load_timeseries:
            self.download_csv_files()
            self.write_index()
            self.write_h5_database_from_csv(self.index)
        if not h5_database_ok and not recompile_h5_from_csv and load_timeseries:
            self.download_h5_database()
            
        self.correct_sequence_length(original_time_serie_lengths)
    
        self.index = pd.read_csv(self.indexfile, index_col=None)
        self.index = self.index.loc[self.index["CODE_CULTU"].isin(self.mapping.index)]
        if verbose:
            print(f"kept {len(self.index)} time series references from applying class mapping")
            
        self.set_labels_names() 

        # filter zero-length time series
        if self.index.index.name != "idx":
            self.index = self.index.loc[self.index.sequencelength > filter_length].set_index("idx")

        self.maxseqlength = int(self.index["sequencelength"].max())

        if not os.path.exists(self.codesfile):
            download_file(CODESURL, self.codesfile)
        self.codes = pd.read_csv(self.codesfile, delimiter=";", index_col=0)

        if preload_ram:
            self.X_list = list()
            with h5py.File(self.h5path, "r") as dataset:
                for idx, row in tqdm(self.index.iterrows(), desc="loading data into RAM", total=len(self.index)):
                    self.X_list.append(np.array(dataset[(row.path)]))
        else:
            self.X_list = None

        self.index.rename(columns={"meanQA60": "meanCLD"}, inplace=True)

        if "classid" not in self.index.columns or "classname" not in self.index.columns or "region" not in self.index.columns:
            # drop fields that are not in the class mapping
            self.index = self.index.loc[self.index["CODE_CULTU"].isin(self.mapping.index)]
            self.index[["classid", "classname"]] = self.index["CODE_CULTU"].apply(lambda code: self.mapping.loc[code])
            self.index["region"] = self.region
            self.index.to_csv(self.indexfile)
        self.get_codes()

    def download_csv_files(self):
        zipped_file = os.path.join(self.root, str(self.year), self.level, f"{self.region}.zip")
        download_file(RAW_CSV_URL[self.year][self.level][self.region], zipped_file)
        unzip(zipped_file, self.csvfolder)
        os.remove(zipped_file)

    def build_folder_structure(self, root, year, level, region):
        """
        folder structure

        <root>
           codes.csv
           classmapping.csv
           <year>
              <region>.shp
              <level>
                 <region>.csv
                 <region>.h5
                 <region>
                     <csv>
                         123123.csv
                         123125.csv
                         ...
        """
        year = str(year)

        os.makedirs(os.path.join(root, year, level, region), exist_ok=True)

        h5path = os.path.join(root, year, level, f"{region}.h5")
        indexfile = os.path.join(root, year, level, region + ".csv")
        codesfile = os.path.join(root, "codes.csv")
        shapefile = os.path.join(root, year, f"{region}.shp")
        classmapping = os.path.join(root, "classmapping.csv")
        csvfolder = os.path.join(root, year, level, region, "csv")

        return h5path, indexfile, codesfile, shapefile, classmapping, csvfolder

    def get_fid(self, idx):
        return self.index[self.index["idx"] == idx].index[0]

    def download_h5_database(self):
        print(f"downloading {self.h5path}.tar.gz")
        download_file(H5_URLs[self.year][self.level][self.region], self.h5path + ".tar.gz", overwrite=True)
        print(f"extracting {self.h5path}.tar.gz to {self.h5path}")
        untar(self.h5path + ".tar.gz")
        print(f"removing {self.h5path}.tar.gz")
        os.remove(self.h5path + ".tar.gz")
        print(f"checking integrity by file size...")
        assert os.path.getsize(self.h5path) == FILESIZES[self.year][self.level][self.region]
        print("ok!")

    def write_h5_database_from_csv(self, index):
        with h5py.File(self.h5path, "w") as dataset:
            for idx, row in tqdm(index.iterrows(), total=len(index), desc=f"writing {self.h5path}"):
                X = self.load(os.path.join(self.root, row.path))
                dataset.create_dataset(row.path, data=X)

    def get_codes(self):
        return self.codes

    def download_geodataframe(self):
        targzfile = os.path.join(os.path.dirname(self.shapefile), self.region + ".tar.gz")
        download_file(SHP_URLs[self.year][self.region], targzfile)
        untar(targzfile)
        os.remove(targzfile)

    def geodataframe(self):

        if not os.path.exists(self.shapefile):
            self.download_geodataframe()

        geodataframe = gpd.GeoDataFrame(self.index.set_index("id"))

        gdf = gpd.read_file(self.shapefile)

        # 2018 shapefile calls ID ID_PARCEL: rename if necessary
        gdf = gdf.rename(columns={"ID_PARCEL": "ID"})

        # copy geometry from shapefile to index file
        geom = gdf.set_index("ID")
        geom.index.name = "id"
        geodataframe["geometry"] = geom["geometry"]
        geodataframe.crs = geom.crs

        return geodataframe.reset_index()

    def load_classmapping(self, classmapping):
        if not os.path.exists(classmapping):
            if self.verbose:
                print(f"no classmapping found at {classmapping}, downloading from {CLASSMAPPINGURL}")
            download_file(CLASSMAPPINGURL, classmapping)
        else:
            if self.verbose:
                print(f"found classmapping at {classmapping}")

        self.mapping = pd.read_csv(classmapping, index_col=0).sort_values(by="id")
        self.mapping = self.mapping.set_index("code")
        self.classes = self.mapping["id"].unique()
        self.classname = self.mapping.groupby("id").first().classname.values
        self.klassenname = self.classname
        self.nclasses = len(self.classes)
        if self.verbose:
            print(f"read {self.nclasses} classes from {classmapping}")

    def load_raw(self, csv_file):
        """['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
               'B8A', 'B9', 'QA10', 'QA20', 'QA60', 'doa', 'label', 'id']"""
        sample = pd.read_csv(os.path.join(self.csvfolder, os.path.basename(csv_file)), index_col=0).dropna()

        # convert datetime to int
        sample["doa"] = pd.to_datetime(sample["doa"]).astype(int)
        sample = sample.groupby(by="doa").first().reset_index()

        return sample

    def load(self, csv_file):
        sample = self.load_raw(csv_file)

        selected_bands = SELECTED_BANDS[self.level]
        X = np.array(sample[selected_bands].values)
        if np.isnan(X).any():
            t_without_nans = np.isnan(X).sum(1) > 0
            X = X[~t_without_nans]

        return X

    def load_culturecode_and_id(self, csv_file):
        sample = self.load_raw(csv_file)
        X = np.array(sample.values)

        if self.level == "L1C":
            cc_index = self.bands.index("label")
        else:
            cc_index = self.bands.index("code_cultu")
        id_index = self.bands.index("id")

        if len(X) > 0:
            field_id = X[0, id_index]
            culture_code = X[0, cc_index]
            return culture_code, field_id

        else:
            return None, None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        row = self.index.iloc[index]

        if self.X_list is None:
            with h5py.File(self.h5path, "r") as dataset:
                X = np.array(dataset[(row.path)])
        else:
            X = self.X_list[index]

        # translate CODE_CULTU to class id
        y = self.mapping.loc[row["CODE_CULTU"]].id

        # npad = self.maxseqlength - X.shape[0]
        # X = np.pad(X, [(0, npad), (0, 0)], 'constant', constant_values=PADDING_VALUE)

        if self.transform is not None:
            X = self.transform(X)
        if self.target_transform is not None:
            y = self.target_transform(y)

        return X, y, row.id

    def write_index(self):
        csv_files = os.listdir(self.csvfolder)
        listcsv_statistics = list()
        i = 1

        for csv_file in tqdm(csv_files):
            if self.level == "L1C":
                cld_index = SELECTED_BANDS["L1C"].index("QA60")
            elif self.level == "L2A":
                cld_index = SELECTED_BANDS["L2A"].index("CLD")

            X = self.load(os.path.join(self.csvfolder, csv_file))
            culturecode, id = self.load_culturecode_and_id(os.path.join(self.csvfolder, csv_file))

            if culturecode is None or id is None:
                continue

            listcsv_statistics.append(
                dict(
                    meanQA60=np.mean(X[:, cld_index]),
                    id=id,
                    CODE_CULTU=culturecode,
                    path=os.path.join(self.csvfolder, f"{id}" + ".csv"),
                    idx=i,
                    sequencelength=len(X)
                )
            )
            i += 1

        self.index = pd.DataFrame(listcsv_statistics)
        self.index.to_csv(self.indexfile)
        
    def correct_sequence_length(self, original_time_serie_lengths):
        """
        if corrected is True, only time series with lengths in "original_time_serie_lengths" will be kept.
        Moreover, small classes (nuts and sunflowers) are removed. 
        To take the previous change into account, target_transform is updated to update the labels. 
        """
        if self.corrected and self.level=="L1C":
            # create a file with only 51 and 102 length time series
            df = pd.read_csv(self.indexfile, index_col=None)
            df = df[df["sequencelength"].isin(original_time_serie_lengths)]
            # remove the small classes: classnames nuts and sunflowers
            df = df[~df['classid'].isin([6, 4])]
            # change the classid from 0 to 7, after removing 4 and 6, knowing that the classes were from 0 to 9. 
            dict_new_classid = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4, 7: 5, 8: 6}
            df['classid'] = df['classid'].replace(dict_new_classid)
            self.indexfile = self.indexfile.replace(".csv", "_corrected.csv")
            assert df['classid'].nunique() == 7
            df.to_csv(self.indexfile, index=False)
            # change self.target_transform to take into account the new classid, i.e. following dict_new_classid
            self.target_transform = lambda y: torch.tensor(dict_new_classid[y], dtype=torch.long)

    def set_labels_names(self):
        unique_samples_classnames = self.index[['classname', 'classid']].drop_duplicates()
        # Sort by 'classid'
        sorted_samples = unique_samples_classnames.sort_values(by='classid')
        # Keep only the 'classname' column
        self.labels_names = sorted_samples['classname'].values


def get_default_transform(level):
    # padded_value = PADDING_VALUE
    sequencelength = 45

    bands = SELECTED_BANDS[level]
    if level == "L1C":
        selected_bands = ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9']
    elif level == "L2A":
        selected_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

    selected_band_idxs = np.array([bands.index(b) for b in selected_bands])

    def transform(x):
        # x = x[x[:, 0] != padded_value, :]  # remove padded values

        # choose selected bands
        x = x[:, selected_band_idxs] * 1e-4  # scale reflectances to 0-1

        # choose with replacement if sequencelength smaller als choose_t
        replace = False if x.shape[0] >= sequencelength else True
        idxs = np.random.choice(x.shape[0], sequencelength, replace=replace)
        idxs.sort()

        x = x[idxs]

        return torch.from_numpy(x).type(torch.FloatTensor)

    return transform


def get_default_target_transform():
    return lambda y: torch.tensor(y, dtype=torch.long)

RAW_CSV_URL = {
    2017: {
        "L1C": {
            "frh01": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L1C/frh01.zip",
            "frh02": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L1C/frh02.zip",
            "frh03": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L1C/frh03.zip",
            "frh04": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L1C/frh04.zip",
        },
        "L2A": {
            "frh01": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L2A/frh01.zip",
            "frh02": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L2A/frh02.zip",
            "frh03": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L2A/frh03.zip",
            "frh04": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L2A/frh04.zip",

        }
    },
    2018: {
        "L1C": {
            "frh01": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2018/L1C/frh01.zip",
            "frh02": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2018/L1C/frh02.zip",
            "frh03": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2018/L1C/frh03.zip",
            "frh04": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2018/L1C/frh04.zip",
        }
    }
}

INDEX_FILE_URLs = {
    2017: {
        "L1C": {
            "frh01": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L1C/frh01.csv",
            "frh02": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L1C/frh02.csv",
            "frh03": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L1C/frh03.csv",
            "frh04": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L1C/frh04.csv",
            "belle-ile": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L1C/belle-ile.csv"
        },
        "L2A": {
            "frh01": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L2A/frh01.csv",
            "frh02": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L2A/frh02.csv",
            "frh03": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L2A/frh03.csv",
            "frh04": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L2A/frh04.csv",
            "belle-ile": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L2A/belle-ile.csv"
        }
    },
    2018: {
            "L1C": {
                "frh01": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2018/L1C/frh01.csv",
                "frh02": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2018/L1C/frh02.csv",
                "frh03": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2018/L1C/frh03.csv",
                "frh04": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2018/L1C/frh04.csv",
            }
        }
}

SHP_URLs = {
    2017: {
        "frh01": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/shp/frh01.tar.gz",
        "frh02": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/shp/frh02.tar.gz",
        "frh03": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/shp/frh03.tar.gz",
        "frh04": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/shp/frh04.tar.gz",
        "belle-ile": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/shp/belle-ile.tar.gz"
    },
    2018: {
        "frh01": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2018/shp/frh01.tar.gz",
        "frh02": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2018/shp/frh02.tar.gz",
        "frh03": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2018/shp/frh03.tar.gz",
        "frh04": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2018/shp/frh04.tar.gz",
    }
}

FILESIZES = {
    2017: {
        "L1C": {
            "frh01": 2559635960,
            "frh02": 2253658856,
            "frh03": 2493572704,
            "frh04": 1555075632,
            "belle-ile": 17038944
        },
        "L2A": {
            "frh01": 987259904,
            "frh02": 803457960,
            "frh03": 890027448,
            "frh04": 639215848,
            "belle-ile": 8037952
        }
    },
    2018: {
        "L1C": {
            "frh01": 9878839310,
            "frh02": 8567550069,
            "frh03": 10196638286,
            "frh04": 2235351576 # 6270634169
        }
    }
}

H5_URLs = {
    2017: {
        "L1C": {
            "frh01": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L1C/frh01.h5.tar.gz",
            "frh02": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L1C/frh02.h5.tar.gz",
            "frh03": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L1C/frh03.h5.tar.gz",
            "frh04": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L1C/frh04.h5.tar.gz",
            "belle-ile": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L1C/belle-ile.h5.tar.gz"
        },
        "L2A": {
            "frh01": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L2A/frh01.h5.tar.gz",
            "frh02": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L2A/frh02.h5.tar.gz",
            "frh03": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L2A/frh03.h5.tar.gz",
            "frh04": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L2A/frh04.h5.tar.gz",
            "belle-ile": "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L2A/belle-ile.h5.tar.gz"
        }
    }
}

# 9-classes used in ISPRS submission
CLASSMAPPINGURL = "https://breizhcrops.s3.eu-central-1.amazonaws.com/classmapping.csv"

# 13-classes used in ICML workshop
CLASSMAPPINGURL_ICML = "https://breizhcrops.s3.eu-central-1.amazonaws.com/classmapping_icml.csv"

CODESURL = "https://breizhcrops.s3.eu-central-1.amazonaws.com/codes.csv"

import os
import sys
import urllib
import zipfile
import tarfile
from tqdm import tqdm


def update_progress(progress):
    barLength = 20  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    text = "\rLoaded: [{0}] {1:.2f}% {2}".format("#" * block + "-" * (barLength - block), progress * 100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


def untar(filepath):
    dirname = os.path.dirname(filepath)
    with tarfile.open(filepath, 'r:gz') as tar:
        #tar.extractall(path=dirname)
	#tar = tarfile.open(tar_file)
        for member in tar.getmembers():
            if member.isreg():  # skip if the TarInfo is not files
                member.name = os.path.basename(member.name) # remove the path by reset it
                tar.extract(member,dirname) # extract

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path, overwrite=False):
    if url is None:
        raise ValueError("download_file: provided url is None!")

    if not os.path.exists(output_path) or overwrite:
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    else:
        print(f"file exists in {output_path}. specify overwrite=True if intended")


def unzip(zipfile_path, target_dir):
    with zipfile.ZipFile(zipfile_path) as zip:
        for zip_info in zip.infolist():
            if zip_info.filename[-1] == '/':
                continue
            zip_info.filename = os.path.basename(zip_info.filename)
            zip.extract(zip_info, target_dir)
