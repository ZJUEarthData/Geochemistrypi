<p style="text-align: center; font-size: 38px; font-weight: bold;">
  Developer Guide for Geochemistry π
</p>




#### Table of Contents

1. [Installation](#installation)
2. [Example](#example)
3. [Data Format](#data)
4. [Use](#use)
5. [Bug Report](#bug)
6. [Advice](#advice)



## 1. Installation <a name="installation"> </a>

Requirements: Python 3.9
**Note**:You must have Python 3.9 installed on your computer to use the software.

### 1.1 Check Environment

You need to ensure that pip is installed.
You can view it by typing instructions on the command line.

```
$ pip --version
```
**Note**: **\$** refers to the command line interface. Don't copy **\$**.  The same as below.

### 1.2 Quick Install

One instruction to download our software on command line, such as Terminal on macOS, CMD on Windows.

```
$ pip install geochemistrypi
```

The beta version runs on MacOS, Windows or Linux. Make sure that your network is stable while downloading.

### 1.3 Advanced Install

It is highly recommended downloading in an isolated virtual python environment, which prevents messing up your system python packages.

**Step 1:** Create an virtual environment

```
$ python -m venv my_virtual_environment_name
```

Note: The only element to change is ***my_virtual_environment_name***. (You need to remember the name of your virtual environment so you can use it next time.)
If ***python*** fails, try with ***python3***.

The instruction above will create an virtual environment on the current directory.

**Step 2:** Activate the virtual environment

(1) Operating System: macOS or Linux

In the same directory, run this instruction:

```
$ source my_virtual_environment_name/bin/activate
```

(2) Operating System: Windows

In the same directory, run this instruction:

```
$ my_virtual_environment_name\Scripts\activate.bat
```

**Step 3:** Download our software

```
$ python -m pip install geochemistrypi
```

Make sure that your network is stable while downloading.

In two methods, if you fail to download, please refer to part 4 \<bug report\> .



## 2. Example<a name="example"> </a>

**Beta version:** It only supports the command to apply data mining techniques to deal with your own data. More algorithms and their related functions will be appended to our software in the following released versions.

**How to run:** After successfully downloading, run this instruction on command line whatever directory it is. It will takes some time to load the software into your CPU and memory when the first time to run the software. Just be patient!!

**Case 1:** Run with built-in data set for testing

```
$ geochemistrypi data-mining
```

**Note**: There are four built-in data sets corresponding to four kinds of model pattern, regression, classification, clustering, decomposition.

**Case 2:** Run with your own data set

```
$ geochemistrypi data-mining --data your_own_data_set.xlsx
```

**Note**: Currently, only `.xlsx` file is supported. Please specify the path your data file exists. The only element to change is ***your_own_data_set.xlsx***.

 If you run the command above, it means the command is executed under the directory your data file exists. Related processed results will be in the same directory.

**Other Commands:**

```
$ geochemistrypi --help
```

It shows the related information of our software, including brief introduction and commands.

```
$ geochemistrypi --install-completion zsh
```

It allows to configure auto-completion function for users' computer. The only element to change is ***zsh***. After relaunching the shell (Terminal on macOS, CMD on Windows). You can use *tab* on your keyboard to implement auto-completion.

```
$ geochemistrypi --show-completion zsh
```

It shows the codes of configuration of auto-completion on your computer. The only element to change is ***zsh***.

```
$ geochemistrypi data-mining --help
```

It shows the infomation of *data-mining* command.


## 3. Data Format<a name="data"> </a>

In order to utilize the functions provided by our software, your own data set should satisfy:

+ be with the suffix ***.xlsx***, which is supported by Microsoft Excel.
+ be comprise of location information ***LATITUDE*** and ***LONGITUDE***, two columns respectively.

If you want to run classification algorithm, only supporting binary classification currently, you data set should satisfy:

+ Tag column ***LABEL*** to differentiate the data

The following are four built-in data set in our software stored on Google Drive, have a look on them.

+ [Data_Regression.xlsx (International - Google drive)](https://docs.google.com/spreadsheets/d/13MB4t_2PiZ90tTMJKw7HcBUi2sb3tXej/edit?usp=sharing&ouid=110717816678586054594&rtpof=true&sd=true)
+ [Data_Regression.xlsx (China - Tencent Docs)](https://docs.qq.com/document/DQ3VmdWZCTGV3bmpM?&u=6868f96d4a384b309036e04e637e367a)

+ [Data_Classification.xlsx (International - Google drive)](https://docs.google.com/spreadsheets/d/1xFBCYVmtZfuEAbeBljUlzqBjxVuLAt8x/edit?usp=sharing&ouid=110717816678586054594&rtpof=true&sd=true)
+ [Data_Classification.xlsx (China - Tencent Docs)](https://docs.qq.com/document/DQ0JUaUFsZnRaZkNG?&u=6868f96d4a384b309036e04e637e367a)

+ [Data_Clustering.xlsx (International - Google drive)](https://docs.google.com/spreadsheets/d/1sbuJdOzGNQ2Pk-bVURfPYg1rltyBbn5J/edit?usp=sharing&ouid=110717816678586054594&rtpof=true&sd=true)
+ [Data_Clustering.xlsx (China - Tencent Docs)](https://docs.qq.com/document/DQ3dKdGtlWkhZS2xR?&u=6868f96d4a384b309036e04e637e367a)

+ [Data_Decomposition.xlsx (International - Google drive)](https://docs.google.com/spreadsheets/d/1kix82qj5--vhnm8-KhuUBH9dqYH6zcY8/edit?usp=sharing&ouid=110717816678586054594&rtpof=true&sd=true)
+ [Data_Decomposition.xlsx (China - Tencent Docs)](https://docs.qq.com/document/DQ29oZ0lhUGtZUmdN?&u=6868f96d4a384b309036e04e637e367a)

## 4. Use<a name="use"> </a>

### 4.1 Selection algorithm
When you type geochemistrypi data-mining or geochemistrypi data-mining --data your_own_data_set.xlsx command,there are four algorithmic modes you can use,which are Regression,Classification,Clustering and Dimensional Reduction.

~~~
Geochemistry Py v.1.0.0 - Beta Version
....... Initializing .......
-*-*- Data Loading -*-*-
Built-in Data Option:
1 - Data For Regression
2 - Data For Classification
3 - Data For Clustering
4 - Data For Dimensional Reduction
(User) ➜ @Number:
~~~
You can choose which one you want to use,just type in its number.
Then you can see a column corresponding to you data.

### 4.2 World Map Projection for A Specific Element Option
You can press Enter to go to the next step.
Here you can choose to project specific elements in the world map.

~~~
World Map Projection for A Specific Element Option:
1 - Yes
2 - No
(Plot) ➜ @Number:
~~~

if you choose yes,it will produce a drawing automatically.
then you can choose one column of your date set,then it will produce a drawing automatically.

~~~
-*-*- Distribution in World Map -*-*-
Select one of the elements below to be projected in the World Map:
--------------------
(Plot) ➜ @Number:
~~~

### 4.3 Do you want to continue to project a new element in the World Map
Here you can choose whether or not to continue to project some elements.

~~~
Do you want to continue to project a new element in the World Map?
1 - Yes
2 - No
(Plot) ➜ @Number:
~~~

### 4.4 Select the data range you want to process
Here is about data selection.
Sometimes we do not need to process all of the data columns,just need to select the columns we want to process according to one input format,such as we need the column from the 8th to 13th,the input format is "[8,13]".

~~~
Select the data range you want to process.
Input format:
Format 1: "[**, **]; **; [**, **]", such as "[1, 3]; 7; [10, 13]" --> you want to deal with the columns 1, 2, 3, 7, 10, 11, 12, 13
Format 2: "xx", such as "7" --> you want to deal with the columns 7
@input:
~~~

Next,it will show the columns information

~~~
--------------------
Index - Column Name
8 - SIO2(WT%)
9 - TIO2(WT%)
10 - AL2O3(WT%)
11 - CR2O3(WT%)
12 - FEOT(WT%)
13 - CAO(WT%)
--------------------
~~~

then it check whether our columns has some missing values.

~~~
The Selected Data Set:
     SIO2(WT%)  TIO2(WT%)  AL2O3(WT%)  CR2O3(WT%)  FEOT(WT%)   CAO(WT%)
0    53.536000   0.291000    3.936000       1.440   3.097000  18.546000
..         ...        ...         ...         ...        ...        ...
108  51.960000   0.550000    6.490000       0.800   2.620000  20.560000

[109 rows x 6 columns]
(Press Enter key to move forward.)
~~~

### 4.5 Strategy for Missing Values
If some columns has some null values,we provide currently three methods to fill the missing values,you can choose whatever you want.

~~~
-*-*- Strategy for Missing Values -*-*-
1 - Mean
2 - Median
3 - Most Frequent
Which strategy do you want to apply?
(Data) ➜ @Number:
~~~

### 4.6 Hypothesis Testing on Imputation Method
After chooseing the imputation method and it will act the hypothesis testing on the imputation method.

### 4.7 Feature Engineering
This is the feature engineering part.

~~~
Feature Engineering Option:
1 - Yes
2 - No
(Data) ➜ @Number:
~~~

If you choose yes,you can create a new feature.
Firstly,you need to name the feature.Then you can do some arithmetic operation on the new feature.

~~~
Selected data set:
a - SIO2(WT%)
b - TIO2(WT%)
c - AL2O3(WT%)
d - CR2O3(WT%)
e - FEOT(WT%)
f - CAO(WT%)
Name the constructed feature (column name):
@input: GEO
~~~

You just follow the format such as "a*b-c",then can create the new feature column based on the columns we provided.

~~~
Build up new feature with the combination of 4 basic arithmatic operator.
Input example 1: a * b - c
--> Step 1: Multiply a column with b column;
--> Step 2: Subtract c from the result of Step 1;
Input example 2: (d + 5 * f) / g
--> Step 1: multiply 5 with f;
--> Step 2: Plus d column with the result of Step 1;
--> Step 3: Divide the result of Step 1 by g;
@input:a * b - c
~~~

You can see the information of the new data set you created and can append the feature.the new feature into the original data set.

```
Name: GEO, Length: 109, dtype: float64
```

### 4.8 Do you want to continue to construct a new feature
You can continue to create more features.

~~~
Do you want to continue to construct a new feature?
1 - Yes
2 - No
(Data) ➜ @Number:
~~~

### 4.9 Mode Options
Choose your algorithm in the mode options.

~~~
-*-*- Mode Options -*-*-
1 - Regression
2 - Classification
3 - Clustering
4 - Dimensional Reduction
(Model) ➜ @Number:
~~~

### 4.10 Data Split - X Set and Y Set
Here you need to divide the data set into a X set and Y set.
Firstly You need to have a feature column.
You can choose one or several columns to be the X set.

~~~
The selected X data set:
Select the data range you want to process.
Input format:
Format 1: "[**, **]; **;", such as "[1, 3]; 7;" --> you want to deal with the columns 1, 2, 3
Format 2: "xx", such as "7" --> you want to deal with the columns 7
@input:[1,5]
~~~

### 4.11 Feature Scaling on X Set
You can do some feature scaling one the X set.
Choose yes in feature scaling on the X set.

~~~
-*-*- Feature Scaling on X Set -*-*-
1 - Yes
2 - No
(Data) ➜ @Number:
~~~

Choose the strategy you want to apply (General use the standardization method).

~~~
Which strategy do you want to apply?
1 - Min-max Scaling
2 - Standardization
(Data) ➜ @Number:
The data range in the dimensional space has changed.
Then you need to create traget column.
~~~

### 4.12 Data Split - Train Set and Test Set
Create train set and set based on the X and Y.

~~~
The selected Y data set:
Note: Normally, only one column is allowed to be tag column, not multiple columns.
Select the data range you want to process.
Input format:
Format 1: "[**, **]; **;", such as "[1, 3]; 7;" --> you want to deal with the columns 1, 2, 3
Format 2: "xx", such as "7" --> you want to deal with the columns 7
@input:
~~~

You need to Choose the ratio to divide them into training set and testing set(Note can tell you how to do).

~~~
-*-*- Data Split - Train Set and Test Set -*-*-
Note: Normally, set 20% of the dataset aside as test set, such as 0.2
(Data) ➜ @Test Ratio: 0.2
~~~

### 4.13 Model Selection
You can choose one in the model selection(Such as 8 is the algorithm to implement deep neural network).

~~~
-*-*- Model Selection -*-*-:
1 - Linear Regression
2 - Polynomial Regression
3 - Support Vector Machine
4 - Decision Tree
5 - Random Forest
6 - Extra-Trees
7 - Xgboost
8 - Deep Neural Networks
9 - All models above to be trained
Which model do you want to apply?(Enter the Corresponding Number)
(Model) ➜ @Number:8
~~~
**Note**:
The following takes deep neural networks an an example.
You can choose whether to run automatic machine learning.
Then following the hint to input.

**Note**:
You can check the result under this directory,it will tell you where those results.
You can also choose to watch video tutorials.[Download and Run the Bata Version](https://www.bilibili.com/video/BV1UM4y1Q7Ju/?spm_id_from=333.999.0.0).



## 5. Bug Report<a name="bug"> </a>

  Due to the problems of dependency management in Python and specific computers configuration, you might encounter some unpredictable issues when downloading our software.

**Issue 1:** Http problem. It is caused when your network is unstable.

Solved: a) Make sure the network works well; b) Try *pip* install instruction shown above several times until it downloads all denpendencies.

**Issue 2:** Dependecy Version. It is caused when not to use virutal environment.

Solved: Download our software in an isolated virtual environment, try with ***advanced install*** above.

**Issue 3:** Software package installation failure.Such as "No module named 'pandas"

Solved:You can install the failed package separately.Just type the following in the terminal

```
$ pip install pandas
```
or
```
$ pip install pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas
```
**pandas** is the module,you only to change it.

**Ways to report other bugs** (downloading or running the software):

1. Contact Sany (Email: sanyhew1097618435@163.com) with the title as "Bug: Geochemistry Pi".
2. Open a GitHub issue and follow the [template](https://github.com/ZJUEarthData/geochemistrypi/issues/26) to detail the problem.

We promise to get you in contact as soon as possible.



## 6. Advice<a name="advice"> </a>

The software is in beta version currently. There are too many shortcomings which we need to improve in the future. It would be highly appreciated if you can share your opinions on how to make it better.

However, considering the tradeoff of time and cost, it is likely that we are unable to perfect parts of the functions in time. Here, we would like to exten our warm invitation to you to join in this open-source software project to make your own contribution to the whole community. You will learn the whole framwork through our specialized training procedure and be able to make customization on your own. For more Information, check our GitHub page (International) or Gitee page (China).

+ [Github - geochemistrypi](https://github.com/ZJUEarthData/geochemistrypi)
+ [Gitee - geochemistrypi](https://gitee.com/zju-earth-data/geochemistrypi)

**Ways to Contact:**

+ Can He (Email: sanyhew1097618435@163.com) with the title as "Advice: Geochemistry Pi".

+ ZJU Earth Data: zhangzhou333@zju.edu.cn
