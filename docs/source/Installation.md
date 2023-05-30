## Installation

Requirements: Python 3.9+

**Note**: You must have Python 3.9+ installed on your computer to use the software.

<br />

### 1.1 Check Environment

You need to ensure that pip is installed.
You can view it by typing instructions on the command line.

```
$ pip --version
```
**Note**: **\$** refers to the command line interface. Don't copy **\$**.  The same as below.

<br />

### 1.2 Install `Geochemistrypi` package
#### 1.2.1 Quick Install

One instruction to download our software on command line, such as Terminal on macOS, CMD on Windows.

```
$ pip install geochemistrypi
```

The beta version runs on MacOS, Windows or Linux. Make sure that your network is stable while downloading.

#### 1.2.2 Advanced Install

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

<br />

### 1.3 How to invoke `Geochemistrypi` package and start data-mining on your terminal

After installing `Geochemistrypi` package into your environment, you can invoke the package and enter the data-mining interface by simply running:

```python
$ geochemistrypi data-mining #loading the testing dataset provided by us
```

Note that if you want to run your own dataset, run

```python
$ geochemistrypi data-mining --data YOURDATA.xlsx #YOURDATA should be replaced by the file name of your dataset
```
