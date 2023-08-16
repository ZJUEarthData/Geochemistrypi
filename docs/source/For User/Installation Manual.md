<p style="text-align: center; font-size: 38px; font-weight: bold;">
Installation Manual
</p>

#### Contents

1. [Preparation](#Preparation)
2. [Download Geochemistry π](#Download-Geichemistry—π)
3. [Solutions and Suggestions for Installation Failure](#Solutions)


## 1. Preparation <a name="Preparation"> </a>

### 1.1 Install Python Interpreter

A Python interpreter is a program that reads and executes Python code. When you write Python code in a text file with a `.py` extension, you can run that file using the Python interpreter.  For example, use `python main.py` to executes the codes inside main.py file in command line or terminal.

The normal ways to install Python interpreter:

(1) If you are a Windows user, you can use Microsoft Store App to download directly by searching Python.

(2) Refer to the download section in [Python official documentation](https://www.python.org).

(3) If you are Chinese users, you can refer to this blog [Python Download - RUNOOB](https://www.runoob.com/python/python-install.html) to download too.

### 1.2  Install Conda

Conda allows you to easily install, update, and manage Python packages and dependencies. Usually,  Conda is included the software Anaconda. Hence, by downloading Anaconda, you can install Conda too.

The normal ways to install Anaconda:

(1) Refer to the download section in [Anaconda website](https://www.anaconda.com).

(2) If you are Chinese users, you can refer to [Anaconda Download - Zhihu](https://zhuanlan.zhihu.com/p/459601766) to download Anaconda using Tsinghua mirror source Anaconda. Also, if you are not familiar with Command Prompt (CMD) in Windows, you can reference to [Frequently Used Commands on Windows - Zhihu](https://zhuanlan.zhihu.com/p/67513308).


## 2. Download Geochemistry π in Virtual Environment <a name="Download-Geichemistry—π"> </a>

### 2.1 Create A Virtual Environment

Use Conda to manage virtual environments (recommended) :

(1) Creates a virtual environment by installing the python interpreter, for example, to install a version 3.9 python interpreter, where `env_name` is the name of the created environment. To avoid version problems, it is better to use 3.9 version of python.

On Mac Terminal:

```
conda create -n vir_env_name python=3.9
```

On Windows Command Prompt:

```
conda create -n vir_env_name python=3.9
```

For the prompting information, input `y` to continue until the configuration is done.

(2) Activate the created virtual environment.

On Mac Terminal:

```
conda activate vir_env_name
```

On Windows Command Prompt:

```
conda activate vir_env_name
```

For more useful Conda commands,  please search online.

### 2.2 Use pip to Download

After the virtual environment is activated on your computer, you can follow the steps below to download our software:

(1) Clear the cache packages:

On Mac Terminal:

```
pip cache purge
```

On Windows Command Prompt:

```
pip cache purge
```

(2) Download our software:

On Mac Terminal:

```
pip install geochemistrypi
```

On Windows Command Prompt:

```
pip install geochemistrypi
```

(3) Check the latest version of our software:

On Mac Terminal:

```
geochemistrypi --version
```

On Windows Command Prompt:

```
geochemistrypi --version
```

**Note**: Domestic direct installation may stop because of network speed problems in ray or Fiona package installation failure. You can reference the following video to resolve the problem.

+ [Possible Scenarios When Installing via pip Directly in China.mp4](https://www.bilibili.com/video/BV1Gs4y1d7Cm/?spm_id_from=333.999.0.0&vd_source=350db2ec0e0c3ee7f424928a21e82674)


## 3. Solutions and Suggestions for Installation Failure <a name="Solutions"> </a>

### 3.1 Use Tsinghua Mirror Source

If you cannot download our software because Ray/Fiona downloads are too slow or just fail , you can use the pip with Tsinghua mirror source to re-download the package in terms of the specific error.
```
pip install ray -i https://pypi.tuna.tsinghua.edu.cn/simple
```
+ Reference video: [Solutions to Failures in Direct pip Installation in China.mp4](https://www.bilibili.com/video/BV1zg4y1j7bx/?spm_id_from=333.999.0.0&vd_source=350db2ec0e0c3ee7f424928a21e82674).

### 3.2 Use 'pip install -r requirements/production.txt'

Another way to download related dependecies is to clone the source code from GitHub or Gitee repository firstly:

GitHub Link: https://github.com/ZJUEarthData/geochemistrypi

Gitee Link: https://gitee.com/zju-earth-data/geochemistrypi

After that, unpacking the source code file. Open Terminal on Mac or Command Promt on Window and navigate to the directory to the source code file, use the following command to download the dependency  :

```
pip install -r requirements/production.txt
```

Or use the Tsinghua mirror source to download:

```
pip install -r requirements/production.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

+ Reference video:  [The Fastest Currently Feasible Installation Method in China—Installing from GitHub Using requirements.mp4](https://www.bilibili.com/video/BV1pM411V7iR/?spm_id_from=333.999.0.0&vd_source=350db2ec0e0c3ee7f424928a21e82674)

**Note**: Actually this method can be used for developers to test our latest updates. For more information, please refer to our online documentation in **Local Deployment** under the section of **FOR DEVELOPER**.

### 3.3 Report An Error to Our Team

You can refer to our online documentation in **Contact Us** under the section of **FOR USER**.
