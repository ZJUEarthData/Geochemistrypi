![Geochemistry π.png](https://github.com/ZJUEarthData/geochemistrypi/assets/66779478/8b8c9a61-68bb-40ca-8545-c96e6802bda5)
<p align="center">
<img src="https://img.shields.io/github/actions/workflow/status/ZJUEarthData/geochemistrypi/geochemistrypy.yml?logo=github">
<img src="https://img.shields.io/github/license/ZJUEarthData/geochemistrypi">
<img src="https://img.shields.io/github/v/release/ZJUEarthData/geochemistrypi?include_prereleases">
<img src="https://static.pepy.tech/personalized-badge/geochemistrypi?period=total&units=international_system&left_color=grey&right_color=green&left_text=Downloads">
</p>

# Introduction

## What it is

Geochemistry π is an **open-sourced highly automated machine learning Python framework** dedicating to build up MLOps level 1 software product for data-driven geochemistry discovery on tabular data.

Core capabilities are:

+ **Continous Training**
+ **Machine Learning Lifecycle Management**
+ **Model Inference**

Key features are:

+ **Easy to use:** The automation of data mining process provides the users with simple number options to choose.
+ **Extensible:** It allows appending new algorithms through Scikit-learn with automatic hyper parameter searching by FLAML and Ray.
+ **Traceable**: It integrates MLflow to build special storage mechanism to streamline the end-to-end machine learning lifecycle.

Latest Update: follow up by clicking `Starred` and  `Watch` on our [GitHub repository](https://github.com/ZJUEarthData/geochemistrypi), then get email notifications of the newest features automatically.

The following figure is the simplified overview of Geochemistry π: <br>

<p align="center">
  <img src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/28e174f0-1f2f-4367-96bd-9526352101bd" alt="Overview of workflow" width="600" />
</p>

The following figure is the frontend-backend separation architecture of Geochemistry: <br>

<p align="center">
  <img src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/3b27cbdb-ff50-4fa6-b1d1-4c75b253fdff" alt="Frontend-backend separation architecture of Geochemistry" width="450" />
</p>


**Cite the work as:**

ZhangZhou J*, He Can*, Sun Jianhao, Zhao Jianming, Lyu Yang, Wang Shengxin, Zhao Wenyu, Li Anzhou, Ji Xiaohui. Geochemistry π: Automated machine learning python framework for tabular data (2024). Geochemistry, Geophysics,
Geosystems, 25, e2023GC011324

Download link: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023GC011324


## In-house Materials
Materials are in both Chinese and English. Others unshown below are internal materials.
1. [Guideline Manual – Geochemistry π (International - Google drive)](https://docs.google.com/document/d/1LjwB5Lazk33E5vbtnFPJio_MyjYQxjEu/edit?usp=sharing&ouid=110717816678586054594&rtpof=true&sd=true)
2. [Guideline Manual – Geochemistry π (China - Tencent Docs)](https://docs.qq.com/doc/DQ21IZUdVQktqRWpm?&u=6868f96d4a384b309036e04e637e367a)
3. [Learning Steps for Newbies – Geochemistry π (International - Google drive)](https://docs.google.com/document/d/1GQO-SXwEx_8midr362pqfxNZtfUf-nA6/edit?usp=sharing&ouid=110717816678586054594&rtpof=true&sd=true)
4. [Learning Steps for Newbies - Geochemistry π (China - Tencent Docs)](https://docs.qq.com/doc/DTlVEakt2WnJrdkN1?&u=6868f96d4a384b309036e04e637e367a)
5. [Code Specification v2.1.2 - Geochemistry π (International - Google drive)](https://drive.google.com/file/d/12UPrGqrj9hl0_vK8r-m6xykh_6052OtI/view?usp=sharing)
6. [Code Specification v2.1.2 - Geochemistry π (China - Tencent Docs)](https://docs.qq.com/pdf/DQ2pmc1l1Z2t3QVFa?&u=6868f96d4a384b309036e04e637e367a)
7. [Cycle Report - Geochemistry π (International - Google drive)](https://drive.google.com/file/d/1JPZoSLcPRqzu6LDvw8wLQkV2GfJoER51/view?usp=sharing)
8. [Cycle Report - Geochemistry π (China - Tencent Docs)](https://docs.qq.com/pdf/DQ25VSGNlbGx4UkFZ?&u=6868f96d4a384b309036e04e637e367a)

## In-house Videos
Technical record videos are on Bilibili and Youtube synchronously while other meeting videos are internal materials.
More Videos will be recorded soon.
1. [ZJU_Earth_Data Introduction (Geochemical Data, Python, Geochemistry π) - Prof. Zhang](https://www.bilibili.com/video/BV1Lf4y1w7EK?spm_id_from=333.999.0.0)
2. [How to Collaborate and Provide Bug Report on Geochemistry π Through GitHub - Can He (Sany)](https://www.youtube.com/watch?v=1DWoEsqsfvQ&list=PLy8hNsI55lvh1UHjhVhqNUj3xPdV9sEiM&index=3)
3. [Geochemistry π - Download and Run the Beta Version](https://www.youtube.com/watch?v=EeVaJ3H7_AU&list=PLy8hNsI55lvh1UHjhVhqNUj3xPdV9sEiM&index=9)
4. [How to Create and Use Virtual Environment on Geochemistry π - Can He (Sany)](https://www.youtube.com/watch?v=4KFi7OXxD-c&list=PLy8hNsI55lvh1UHjhVhqNUj3xPdV9sEiM&index=4)
5. [How to use Github-Desktop in conflict resolution - Qiuhao Zhao (Brad)](https://www.youtube.com/watch?v=KT1g5JpuUVI&list=PLy8hNsI55lvh1UHjhVhqNUj3xPdV9sEiM)
6. [Virtual Environment & Packages On Windows - Jianming Zhao (Jamie)](https://www.youtube.com/watch?v=e4VqSBuNp_o&list=PLy8hNsI55lvh1UHjhVhqNUj3xPdV9sEiM&index=2)
7. [Git Workflow & Coordinating Synchronization - Jianming Zhao (Jamie)](https://www.bilibili.com/video/BV1Sa4y1f74k?spm_id_from=333.999.0.0&vd_source=9adcf2c5fdeffe1d11c89d441ef598ba)
