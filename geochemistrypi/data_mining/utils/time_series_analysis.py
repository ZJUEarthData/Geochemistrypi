# time_series_analysis

import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# read Excel data
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Choose your Excel dataset", filetypes=[("Excel", "*.xlsx *.xls")])
if not file_path:
    print("finished")
    exit()

# input time interval
bin_size = input("Time interval (Ma)=")
bin_width = float(bin_size)

df = pd.read_excel(file_path, sheet_name=0)

age = df.R_AGE
ageMax = df.R_MAX_AGE
age_error = np.abs(ageMax - age) / 2

# each sample's subaerial probability
x = df.SBAP
Lat = df.LATITUDE
Lon = df.LONGITUDE

# resampling times
iter = 100

all = range(np.size(age))
print(np.size(all))

# random seed
np.random.seed(2025)

# calculate weight
print("Start calculating weights，waiting...")
wei = np.ones((np.size(all), 1))
WEI = np.ones((np.size(all), 1))

# batch processing
batch_size = 2000

# transform Series into numpy array
lat_arr = Lat.values
lon_arr = Lon.values
age_arr = age.values
x_arr = x.values

# set age limit
data_max = np.nanmax(age_arr)
total_age_limit = np.ceil(data_max / bin_width) * bin_width
num_bins = int(total_age_limit / bin_width)

# ensure current range
for i in range(0, np.size(all), batch_size):
    end = min(i + batch_size, np.size(all))

    # current data
    outlat = lat_arr[i:end, np.newaxis]
    outlon = lon_arr[i:end, np.newaxis]
    outage = age_arr[i:end, np.newaxis]

    ka = 1 / (((lat_arr - outlat) / 2) ** 2 + ((lon_arr - outlon) / 2) ** 2 + 1)
    kb = 1 / (((age_arr - outage) / 38) ** 2 + 1)

    a = np.nansum(ka + kb, axis=1)
    WEI[i:end, 0] = 1 / (a / 0.2)

    # deal with NaN
    nan_mask = np.isnan(x_arr[i:end])
    WEI[i:end, 0][nan_mask] = 0

    print(f"Progress: {end}/{np.size(all)}")

# delate some data
index_wei = np.where((np.isinf(WEI[:, 0])) | (np.isnan(x_arr)))[0]
del_index = index_wei

mask = (~np.isinf(WEI[:, 0])) & (~np.isnan(x_arr)) & (WEI[:, 0] > 0)
del_age = age_arr[mask]
del_age_error = age_error.values[mask]
del_p = x_arr[mask]
del_WEI = WEI[mask, 0]
del_WEIP = del_WEI / np.nansum(del_WEI)

# resampling processes
boot6 = np.ones((num_bins, iter))
bootfixa = np.zeros((np.size(del_index), 1))
bootfixy = np.zeros((np.size(del_index), 1))

if np.size(del_index) > 0:
    bootfixy[:, 0] = x_arr[del_index]
print(f"Start bootstrap {iter} times...")

for i in range(iter):
    if np.size(del_index) > 0:
        bootfixa[:, 0] = np.random.normal(loc=age_arr[del_index], scale=age_error.values[del_index])

    # resampling a dataset that is proportional to samples' weights
    bootstrapSamples = np.random.choice(range(np.size(del_age)), size=np.size(del_age), p=del_WEIP)

    boot1 = np.random.normal(loc=del_age[bootstrapSamples], scale=del_age_error[bootstrapSamples]).reshape(-1, 1)
    boot2 = del_p[bootstrapSamples].reshape(-1, 1)

    bootage_cmb = np.vstack((bootfixa, boot1))
    booty_cmb = np.vstack((bootfixy, boot2))

    boot3 = np.hstack((bootage_cmb, booty_cmb))
    boot4 = boot3[boot3[:, 0].argsort()]

    boot5_list = []
    for L in range(1, num_bins + 1):
        condition = (boot4[:, 0] >= (L - 1) * bin_width) & (boot4[:, 0] <= L * bin_width)
        Bin = boot4[condition, 1]

        if Bin.size > 0:
            Ave = np.sum(Bin >= 0.5) / Bin.size * 100
        else:
            Ave = np.nan

        boot5_list.append(Ave)

    boot6[:, i] = boot5_list
    if (i + 1) % 10 == 0:
        print(f"Progress: {i + 1}/{iter}")

# plot the subaerial proportion
ave_bin = np.nanmean(boot6, axis=1)[:num_bins]
std_bin = 2 * np.nanstd(boot6, axis=1)[:num_bins]
age_x = np.arange(bin_width / 2, total_age_limit, bin_width)

plt.errorbar(age_x, ave_bin, yerr=std_bin, ecolor="r", capsize=4)
plt.xlabel("Age (Ma)")
plt.ylabel("Subaerial proportion (%)")
plt.xlim((0, 4000))
plt.ylim((0, 100))

plt.savefig("Subaerial proportion.pdf", dpi=500)
