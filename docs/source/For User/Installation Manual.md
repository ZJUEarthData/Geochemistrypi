<p style="text-align: center; font-size: 38px; font-weight: bold;">
Installation Manual
</p>





#### Contents

1. [Preparation](#Preparation)
2. [Download Geochemistry π](#Download-Geichemistry——π)
3. [Solutions and suggestions for installation failure](#Solutions)




## 1. Preparation <a name="Preparation"> </a>

### 1.1 Install python environment on computer
（1）Python download address is [https://www.python.org](https://www.python.org/),and the installation tutorial can be found at [https://www.runoob.com/python/python-install.html](https://www.runoob.com/python/python-install.html).

（2）Anaconda download address is [https://www.anaconda.com](https://www.anaconda.com/),suggest domestic users to refer to [https://zhuanlan.zhihu.com/p/459601766](https://zhuanlan.zhihu.com/p/459601766)Tutorial to download Anaconda using Tsinghua mirror source Anaconda.

（3）Familiar with the use of Command Prompt (CMD) under window system,Reference to [https://zhuanlan.zhihu.com](https://zhuanlan.zhihu.com/p/67513308).


## 2. Download Geochemistry π <a name="Download-Geichemistry——π"> </a>

### 2.1 Use pip to download directly

(1) With the assurance that python/Anaconda is installed on your computer,Python is run ***(command 1)*** directly using CMD.Anaconda requires creating a virtual environment with a python interpreter,using the commands ***(command 2)*** or ***(command 3)*** .
```
geochemistrypi --show-completion zsh                                                      (command 1)
```
```
conda install geochemistrypi                                                              (command 2)
or
pip install geochemistrypi                                                                (command 3)
```
**Note:** Anaconda creates a virtual environment by installing the python interpreter, for example, to install a version 3.9 python interpreter,the full command is ***(command 4)*** ,where env_name is the name of the created environment. To avoid version problems, it is better to use 3.9 version of python.
```
conda create -n env_name python=3.9                                                       (command 4)
```
(2) Domestic direct installation may occur because of network speed problems in ray, Fiona package installation failure, the specific installation may occur when the scenario reference video.

+ [Possible Scenarios When Installing via pip Directly in China.mp4](https://www.bilibili.com/video/BV1Gs4y1d7Cm/?spm_id_from=333.999.0.0&vd_source=350db2ec0e0c3ee7f424928a21e82674)

### 2.2 Download with 'pip install -r requirements.txt'

Need to clone the code in advance from the Geochemistry π project, GitHub homepage at [https://github.com/ZJUEarthData/geochemistrypi].
After that, unpacking the file, open the folder, under windows, type CMD in the top left address bar, open CMD, import ***(command 5)*** . Users also could use the Tsinghua mirror source to download, which need changing the command to  ***(command 6)*** .
```
pip install -r requirements.txt                                                           (command 5)
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple               (command 6)
```
+ The installation process for this method can be found in the video  [The Fastest Currently Feasible Installation Method in China—Installing from GitHub Using requirements.mp4](https://www.bilibili.com/video/BV1pM411V7iR/?spm_id_from=333.999.0.0&vd_source=350db2ec0e0c3ee7f424928a21e82674)

## 3. Solutions and suggestions for installation failure <a name="Solutions"> </a>

### 3.1 Ray/Fiona downloads are too slow or fail
If you fail to download the package, you can use the pip command ***(command 7)*** and the Tsinghua mirror source to re-download the package corresponding to the reported error.
```
pip install xxx -i https://pypi.tuna.tsinghua.edu.cn/simple                               (command 7)
```
+ Reference video [Solutions to Failures in Direct pip Installation in China.mp4](https://www.bilibili.com/video/BV1zg4y1j7bx/?spm_id_from=333.999.0.0&vd_source=350db2ec0e0c3ee7f424928a21e82674).

### 3.2 Report an error

Use ***(command 8)*** just reinstall the packages that failed to install.
```
pip install –upgrade xxx –user -i https://pypi.tuna.tsinghua.edu.cn/simple                (command 8)
```
+ Reference video [Solutions to Failures in Direct pip Installation in China.mp4](https://www.bilibili.com/video/BV1zg4y1j7bx/?spm_id_from=333.999.0.0&vd_source=350db2ec0e0c3ee7f424928a21e82674)

### 3.3 Use of Tsinghua Mirror Source

The Tsinghua source image solves the problem of slow pip installation. Simply ***(command 9)*** when using pip and you're good to go.
```
pip install-i https://pypi.tuna.tsinghua.edu.cn/simple                                    (command 9)
```
