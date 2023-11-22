<p>
<img src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/6f970708-7811-40bb-8172-9b98563c7659" class="center"/>
</p>
<p align="center">
<img src="https://img.shields.io/github/actions/workflow/status/ZJUEarthData/geochemistrypi/geochemistrypi.yml?logo=github">
<img src="https://img.shields.io/github/license/ZJUEarthData/geochemistrypi">
<img src="https://img.shields.io/github/v/release/ZJUEarthData/geochemistrypi?include_prereleases">
<img src="https://static.pepy.tech/personalized-badge/geochemistrypi?period=total&units=international_system&left_color=grey&right_color=green&left_text=Downloads">
<img src="https://img.shields.io/pypi/pyversions/geochemistrypi">
</p>

---
**Documentation**: <a href="https://geochemistrypi.readthedocs.io" target="_blank">https://geochemistrypi.readthedocs.io</a>

**Source Code**: <a href="https://github.com/ZJUEarthData/geochemistrypi" target="_blank">https://github.com/ZJUEarthData/geochemistrypi</a>
___

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

## Quick Installation

One instruction to download on **command line**, such as Terminal on macOS, Command Prompt on Windows.
```
pip install geochemistrypi
```
One instruction to download on **Jupyter Notebook** or **Google Colab**.
```
!pip install geochemistrypi
```
**Note**: For more detail on installation, please refer to our online documentation in **Installation Manual** under the section of **FOR USER**. Over there, we highly recommend to use virtual environment (Conda) to avoid dependency version problems.


## Quick Update

One instruction to update the software to the latest version on **command line**, such as Terminal on macOS, Command Prompt on Windows.
```
pip install --upgrade geochemistrypi
```
One instruction to download on **Jupyter Notebook** or **Google Colab**.
```
!pip install --upgrade geochemistrypi
```
Check the latest version of our software:
```
geochemistrypi --version
```



## Example

**How to run:** After successfully downloading, run this instruction on **command line / Jupyter Notebook / Google Colab** whatever directory it is.

### Case 1: Run with built-in data set for testing

On command line:
```
geochemistrypi data-mining
```
On Jupyter Notebook / Google Colab:
```
!geochemistrypi data-mining
```
**Note**: There are four built-in data sets corresponding to four kinds of model pattern.

### Case 2: Run with your own data set

On command line:
```
geochemistrypi data-mining --data your_own_data_set.xlsx
```
On Jupyter Notebook / Google Colab:
```
!geochemistrypi data-mining --data your_own_data_set.xlsx
```
**Note**: Currently, only `.xlsx` file is supported. Please specify the path your data file exists. For Google Colab, don't forget to upload your dataset first.

### Case 3: Activate MLflow web interface

On command line:

```
geochemistrypi data-mining --mlflow
```

On Jupyter Notebook / Google Colab:

```
!geochemistrypi data-mining --mlflow
```
**Note**: Once you run our software, there are two folders (`geopi_output` and `geopi_tracking`) generated automatically. Make sure the directory where you execute the above command should have the genereted file `geopi_tracking`.

Copy the address in the displayed result into any browser to open the MLflow web interface. The address is normally like this http://127.0.0.1:5000. Search MLflow online to see more operations and usages.

For more details: Please refer to:

+ [Manual v1.1.0 for Geochemistry π - Beta (International - Google drive)](https://drive.google.com/file/d/1yryykCyWKM-Sj88fOYbOba6QkB_fu2ws/view?usp=sharing)
+ [Manual v1.1.0 for Geochemistry π - Beta (China - Tencent Docs)](https://docs.qq.com/pdf/DQ0l5d2xVd2VwcnVW?&u=6868f96d4a384b309036e04e637e367a)
+ [Geochemistry π - Download and Run the Beta Version (International - Youtube)](https://www.youtube.com/watch?v=EeVaJ3H7_AU&list=PLy8hNsI55lvh1UHjhVhqNUj3xPdV9sEiM&index=9)
+ [Geochemistry π - Download and Run the Beta Version (China - Bilibili)](https://www.bilibili.com/video/BV1UM4y1Q7Ju/?spm_id_from=333.999.0.0&vd_source=27944ab3b73a78970c1a52a5dcbb9140)



## Roadmap

### First Phase

It works as a **software application** with a command-line interface (CLI) to automate **data mining** process with frequently-used **machine learning algorithms** and **statistical analysis methods**, which would further lower the threshold for the geochemists.

The highlight is that through choosing **simple number options**, the users are able to implement a full cycle of data mining **without knowledge of** SciPy, NumPy, Pandas, Scikit-learn, FLAML, Ray packages.

The following figure is the activity diagram of automated ML pipeline in Geochemistry π:

<img src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/b7f8e6bc-0c6c-4917-adfa-f00c6ccbad7b" />

Its data section provides feature engineering based on **arithmatic operation**. It allows the users to have a statistic analysis on the data set as well as on the imputation result, which is supported by the combination of **Monte Carlo simulation** and **hypothesis testing**.

Its models section provides both **supervised learning** and **unsupervised learning** methods from **Scikit-learn** framework, including four types of algorithms, regression, classification, clustering, and dimensional reduction. Integrated with **FLAML** and **Ray** framework, it allows the users to run AutoML easily, fastly and cost-effectively on the built-in supervised learning algorithms in our framework.

The following figure is the hierarchical architecture of Geochemistry π:
<p align="center">
  <img src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/9c3ddc2b-700c-4685-b52f-f5f9a8931849" alt="Hierarchical Architecture" width="450" />
</p>

### Second Phase

Currently, we are building three access ways to provide more user-friendly service, including **web portal**, **CLI package** and **API**. It allows the user to perform **continuous training** and **model inference** by automating the ML pipeline and **machine learning lifecycle management** by unique storage mechanism in different access layers.

The following figure is the system architecture diagram: <br>

![System Architecture Diagram](https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/20b5a2a4-f2de-492d-a2df-9282196d8c4f)


The following figure is the customized automated ML pipeline: <br>

<p align="center">
  <img src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/07078b43-30bd-46cf-abad-2da509fae6aa" alt="Customized automated ML pipeline" width="400" />
</p>

The following figure is the design pattern hierarchical architecture: <br>

![Design Pattern](https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/aa84ab12-c95e-4282-a60e-64ba2858c437)
![Workflow Object](https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/f08885bf-1bec-4045-bf6b-82c5c18d3f8f)

The following figure is the storage mechanism: <br>

<p align="center">
  <img src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/401f3429-c44f-4b76-b085-7a9dcc987cde" alt="Storage Mechanism" width="500" />
</p>

The whole package is under construction and the documentation is progressively evolving.



## Team Info
**Leader:**
+ Can He (Sany, National University of Singapore, Singapore)
  Email: sanyhew1097618435@163.com

**Technical Group:**
+ Jianming Zhao (Jamie, Jilin University, Changchun, China)
+ Jianhao Sun (Jin, China University of Geosciences, Wuhan, China)
+ Kaixin Zheng (Hayne, Sun Yat-sen University, China)
+ Jianing Wang (National University of Singapore, Singapore)
+ Yongkang Chan (Kill-virus, Lanzhou University, China)
+ Mengying Ye (Mary, Jilin University, China)
+ Mengqi Gao (China University of Geosciences, Beijing, China)

**Product Group**:
+ Yang Lyu (Daisy, Zhejiang University, China)
+ Wenyu Zhao (Molly, Zhejiang University, China)
+ Keran Li (Kirk, Chengdu University of Technology, China)
+ Aixiwake·Janganuer (Ayshuak, Sun Yat-sen University, China)
+ Bailun Jiang (EPSI / Lille University, France)
+ Yucheng Yan (Andy, University of Sydney, Australia)
+ Ruitao Chang (China University of Geosciences Beijing, China)
+ Zhenglin Xu (Garry, Jilin University, China)
+ Junchi Liao(Roceda, University of Electronic Science and Technology of China, China)



## Join Us :)

**The recruitment of research interns is ongoing !!!**

**Key Point: All things are done online, remote work (\*^▽^\*)**

**What can you learn?**
+ Learning the full cycle of data mining on tabular data, including the algorithms in regression,classification, clustering, and decomposition.
+ Learning to be a qualified Python developer, including any Python programing contents towards data mining, basic software engineering techniques like frontend and backend development framework, and cooperation tools like Git.

**What can you get?**

+ Research internship proof and reference letter after working for >> 100 hours.
+ Chance to pay a visit to Hangzhou, China, sponsored by ZJU Earth Data.
+ Chance to be guided by the experts from IT companies in Silicon Valley and Hangzhou.
+ Bonus depending on your performance.

**Current Working Pattern:**
+ Online working and cooperation
+ Three weeks per working cycle -> One online meeting per working cycle
+ One cycle report (see below) per cycle - 5 mins to finish

Even if you are not familiar with topics above, but if you are interested in and have plenty of time to do it. That's enough. We have a full-developed training system to help you, as a newbie of data mining or Python developer, learn steps by steps with seniors until you can make a significant contribution to our project.

**More details about the project?**
Please refer to:
English Page: https://person.zju.edu.cn/en/zhangzhou
Chinese Page: https://person.zju.edu.cn/zhangzhou#0

**Do you want to contribute to this open-source program?**
Contact with your CV: sanyhew1097618435@163.com



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



## Contributors

+ Shengxin Wang (Samson, Lanzhou University, China)
+ Qiuhao Zhao (Brad, Zhejiang University, China)
+ Anzhou Li (Andrian, Zhejiang University, China)
+ Dan Hu (Notre Dame University, United States)
+ Xunxin Liu (Tante, China University of Geosciences, Wuhan, China)
+ Fang Li (liv, Shenzhen University, China)
+ Xin Li (The University of Manchester, United Kingdom)
+ Ting Liu (Kira, Sun Yat-sen University, China)
+ Xirui Zhu (Rae, University of York, United Kingdom)
