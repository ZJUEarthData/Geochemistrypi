<img src="./docs/Geochemistry π.png" width="50%"/>

Geochemistry π is **a Python framework** for data-driven geochemistry discovery. It provides an extendable tool and
one-stop shop for **geochemical data analysis** on tabular data. The goal of the Geochemistry π is to create
a series of user-friendly and extensible products of high automation for the full cycle of geochemistry research.  

## Quick Installation

One instruction to download on command line, such as terminal on MacOS, docs on Windows.  
```
pip install geochemistrypi
```
**Note**: The beta version runs on MacOS, Windows or Linux. Make sure that your network is stable while downloading.

It is highly recommended downloading our software in an isolated virtual python environment, which prevents messing up 
your system python packages. Please search `Virtualenv` on Google for more information.

## Example

**How to run:** After successfully downloading, run this instruction on command line whatever directory it is.

### Case 1: Run with built-in data set for testing
```
geochemistrypi data-mining 
```
**Note**: There are four built-in data sets corresponding to four kinds of model pattern, regression, classification,
clustering, decomposition.

### Case 2: Run with your own data set
```
geochemistrypi data-mining --data your_own_data_set.xlsx
```
**Note**: Currently, only `.xlsx` file is supported. Please specify the path your data file exists. If you run the
command above, it means the command is executed under the directory your data file exists.


## First Phase
It works as a **software application** with a command-line interface (CLI) to automate **data mining** process with
frequently-used **machine learning algorithms** and **statistical analysis methods**, which would further lower the
threshold for the geochemists.

The highlight is that through choosing **simple number options**, the users are able to implement a completed cycle of data
mining **without knowledge of** SciPy, NumPy, Pandas, Scikit-learn, FLAML, Ray packages.

Its data section, shown as below, provides feature engineering based on **arithmatic operation**. It allows the users
to have a statistic analysis on the data set as well as on the imputation result, which is supported by the combination
of **Monte Carlo simulation** and **hypothesis testing**.


Its models section provides both **supervised learning** and **unsupervised learning** methods from
**Scikit-learn** framework, including four types of algorithms, regression, classification,
clustering, and dimensional reduction. Integrated with **FLAML** and **Ray** framework, it allows the users to run
AutoML easily, fastly and cost-effectively on the built-in supervised learning algorithms in our framework.

The activity diagram of the Geochemistry π Version 1.0.0:

<img src="./docs/Geochemistryπ-Activity%20Diagram_v1.png" />

The whole package is under construction and the documentation is progressively evolving. 



## Team Info
**Leader:**
+ Can He (Sany, National University of Singapore, Singapore)    
  Email: sanyhew1097618435@163.com

**Core Developers:**
+ Yang Lyu (Daisy, Zhejiang University, China)
+ Jianming Zhao (Jamie, Jilin University, Changchun, China)
+ Jianhao Sun (Jin, China University of Geosciences，Wuhan, China)
+ Shengxin Wang (Samson, Lanzhou University, China)

**Members**:
+ Wenyu Zhao (Molly, Zhejiang University, China)
+ Fang Li (liv, Shenzhen University, China)
+ Ting Liu (Kira, Sun Yat-sen University, China)
+ Kaixin Zheng (Hayne, Sun Yat-sen University, China)
+ Aixiwake·Janganuer (Ayshuak, Sun Yat-sen University, China)
+ Parnanjan Dutta (Presidency University, Kolkata, India)
+ Bailun Jiang (EPSI / Lille University, France)
+ Yongkang Chang (Kill-virus, Langzhou University, China)
+ Xirui Zhu (Rae, University of York, United Kingdom)

## Join Us :)
**The recruitment of research interns is ongoing !!!**

**Key Point: All things are done online, remote work (\*^▽^\*)**

**What can you learn?**
+ Learning the full cycle of data mining on tabular data, including the algorithms in regression,
classification, clustering, and decomposition.
+ Learning to be a qualified Python developer, including any Python programing contents towards data mining,
basic software engineering techniques like OOP developing, and cooperation tools like Git.

**What can you get?**  

+ Research internship proof and reference letter after working for > 200 hours.
+ Chance to pay a visit to Hangzhou, China, sponsored by ZJU Earth Data.
+ Chance to be guided by the experts from IT companies in Silicon Valley and Hangzhou.
+ Bonus depending on your performance. 

**Current Working Pattern:**
+ Online working and cooperation
+ Three weeks per working cycle -> One online meeting per working cycle
+ One cycle report (see below) per cycle - 5 mins to finish

Even if you are not familiar with topics above, but if you are interested in and have plenty of time to do it.
That's enough. We have a full-developed training system to help you, as a newbie of data mining or Python developer,
learn steps by steps with seniors until you can make a significant contribution to our project.

**More details about the project?**  
Please refer to:   
English Page: https://person.zju.edu.cn/en/zhangzhou  
Chinese Page: https://person.zju.edu.cn/zhangzhou#0  

**Do you want to contribute to this open-source program?**   
Contact with your CV: sanyhew1097618435@163.com  

## In-house Materials
Materials are in both Chinese and English. Others unshown below are internal materials.
1. [Guideline Manual – Geochemistry π](https://docs.google.com/document/d/1LjwB5Lazk33E5vbtnFPJio_MyjYQxjEu/edit?usp=sharing&ouid=110717816678586054594&rtpof=true&sd=true)
2. [Learning Steps for Newbies – Geochemistry π](https://docs.google.com/document/d/1GQO-SXwEx_8midr362pqfxNZtfUf-nA6/edit?usp=sharing&ouid=110717816678586054594&rtpof=true&sd=true)
3. [Code Specification v2.1.2 - Geochemistry π](https://drive.google.com/file/d/12UPrGqrj9hl0_vK8r-m6xykh_6052OtI/view?usp=sharing)
4. [Cycle Report - Geochemistry π](https://drive.google.com/file/d/1JPZoSLcPRqzu6LDvw8wLQkV2GfJoER51/view?usp=sharing)

## In-house Videos
Technical record videos are on Bilibili and Youtube synchronously while other meeting videos are internal materials.
1. [ZJU_Earth_Data Introduction (Geochemical Data, Python, Geochemistry π) - Prof. Zhang](https://www.bilibili.com/video/BV1Lf4y1w7EK?spm_id_from=333.999.0.0)
2. [How to Collaborate and Provide Bug Report on Geochemistry π Through GitHub - Can He (Sany)](https://www.youtube.com/watch?v=1DWoEsqsfvQ&list=PLy8hNsI55lvh1UHjhVhqNUj3xPdV9sEiM&index=3)
3. [How to Run Geochemistry π v1.0.0-alpha - Can He (Sany)](https://www.bilibili.com/video/BV1i541117dd?spm_id_from=333.999.0.0)
4. [How to Create and Use Virtual Environment on Geochemistry π - Can He (Sany)](https://www.youtube.com/watch?v=4KFi7OXxD-c&list=PLy8hNsI55lvh1UHjhVhqNUj3xPdV9sEiM&index=4)
5. [How to use Github-Desktop in conflict resolution - Qiuhao Zhao (Brad)](https://www.youtube.com/watch?v=KT1g5JpuUVI&list=PLy8hNsI55lvh1UHjhVhqNUj3xPdV9sEiM)
6. [Virtual Environment & Packages On Windows - Jianming Zhao (Jamie)](https://www.youtube.com/watch?v=e4VqSBuNp_o&list=PLy8hNsI55lvh1UHjhVhqNUj3xPdV9sEiM&index=2)
7. [Git Workflow & Coordinating Synchronization - Jianming Zhao (Jamie)](https://www.bilibili.com/video/BV1Sa4y1f74k?spm_id_from=333.999.0.0&vd_source=9adcf2c5fdeffe1d11c89d441ef598ba)


## Contributors
+ Qiuhao Zhao (Brad, Zhejiang University, China)
+ Anzhou Li (Andrian, Zhejiang University, China) 
+ Xunxin Liu (Tante, China University of Geosciences, Wuhan, China)
+ Xin Li (The University of Manchester, United Kingdom)