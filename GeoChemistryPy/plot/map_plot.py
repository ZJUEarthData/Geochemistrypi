# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

# TODO: check how to download basemap
def map(col, df):
    LAT1=df['LATITUDE (MIN.)']+df['LATITUDE (MAX.)']
    LON1=df['LONGITUDE (MIN.)']+df['LONGITUDE (MAX.)']
    Lat=LAT1/2
    Lon=LON1/2
    #读取并处理经纬度
    M=df['AL2O3(WT%)']
    #读取可视化属性值
    fig=plt.figure(figsize=(24,16),dpi=300)
    # 绘制画板(figsize 设置图形的大小，50为图形的宽，30为图形的高，单位为英寸,dpi 为设置图形每英寸的点数600)
    plt.rcParams['font.sans-serif'] = 'Arial'
    #使用rc配置文件来自定义图形的各种默认属性,设置字体为 'Arial'
    m = Basemap(projection = 'robin',lat_0=0,lon_0=0)
    # 实例化一个map ，投影方式为‘robin’
    m.drawcoastlines()
    # 画海岸线
    m.drawcountries()
    #画国界线
    m.drawmapboundary(fill_color='white')
    #画大洲，颜色填充为白色
    parallels = np.arange(-90., 90., 45.)
    m.drawparallels(parallels,labels=[True,True, True, False],fontsize=30)
    # 标签=(左,右,上,下)
    # 这两行画纬度，范围为[-90,90]间隔为60，标签标记在右、上边
    meridians = np.arange(-180., 180., 60.)
    m.drawmeridians(meridians,labels=[True, False, True, True],fontsize=30)
    # 这两行画经度，范围为[-180,180]间隔为90,标签标记为左、下边
    lon, lat = m(Lon,Lat)
    # lon, lat为给定的经纬度，可以使单个的，也可以是列表
    m.scatter(lon, lat,c=M,edgecolor='grey',marker='D',linewidths=0.5,vmax=3,vmin=0, s=25, alpha=0.3,cmap='BuPu')
    #标注出所在的点，s为点的大小，还可以选择点的性状和颜色等属性 vmax=1200,vmin=0,vmax=300,vmin=0，
    #cmap: colormap，用于表示从第一个点开始到最后一个点之间颜色渐进变化；
    #alpha:  设置标记的颜色透明度
    #linewidths:  设置标记边框的宽度值
    #edgecolors:  设置标记边框的颜色
    cb=m.colorbar(pad=1)
    cb.ax.tick_params(labelsize=30)
    #设置色标刻度字体大小。
    cb.set_label('Counts',fontsize=50)
    plt.show()

