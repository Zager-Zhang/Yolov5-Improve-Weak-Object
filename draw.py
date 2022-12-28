from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

dir_path = 'CSV'
title_name = 'Loss'
file_name1 = 'results-ECA.csv'  # 3
file_name2 = 'results.csv'  # eca
file_name3 = 'results-ECA+SPP.csv'  # v5
file_name4 = 'results-SPP.csv'  # spp
usecol = [0, 1]
if title_name == 'mAP':
    usecol = [0, 6]
elif title_name == 'Loss':
    usecol = [0, 1]
step, map1 = np.loadtxt(f'{dir_path}/{file_name1}', unpack=True, delimiter=',', skiprows=1, usecols=usecol)
step, map2 = np.loadtxt(f'{dir_path}/{file_name2}', unpack=True, delimiter=',', skiprows=1, usecols=usecol)
step, map3 = np.loadtxt(f'{dir_path}/{file_name3}', unpack=True, delimiter=',', skiprows=1, usecols=usecol)
step, map4 = np.loadtxt(f'{dir_path}/{file_name4}', unpack=True, delimiter=',', skiprows=1, usecols=usecol)

xnew = np.linspace(0, 80, 300)
map1_nb = make_interp_spline(step, map1)(xnew)
map2_nb = make_interp_spline(step, map2)(xnew)
map3_nb = make_interp_spline(step, map3)(xnew)
map4_nb = make_interp_spline(step, map4)(xnew)

fig, ax = plt.subplots()
# plt.ylim(0.6, 0.75)
ax.grid()
ax.plot(xnew, map3_nb, label='yolov5')
ax.plot(xnew, map4_nb, label='yolov5+SPPF-Improve')
ax.plot(xnew, map1_nb, label='yolov5+Attention')
ax.plot(xnew, map2_nb, label='yolov5+Attention+SPPF-Improve')

ax.legend()
ax.set_title(f'{title_name}')
plt.savefig(f'picture/{title_name}')
plt.show()
