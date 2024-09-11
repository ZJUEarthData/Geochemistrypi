# Frequently Asked Questions

**For your reference, we have summarized some problems encountered and solved in the process of development and testing**

**Ways to Contact:**

+ Can He (Email: sanyhew1097618435@163.com) with the title as "Advice: Geochemistry Pi".

+ ZJU Earth Data: zhangzhou333@zju.edu.cn

**Q1. Does it must use the virtual environment?**

Not using a virtual environment may cause conflicts with other code. If you don't use virtual environment, it will overwrite your original dependency package with our specified version, which may affect you to run your other code, similar to the version we specify for pandas download.

In the current beta version, we use pip to do distribution and dependency package management, so all the code you download via pip will be in the specified python (system or virtual) dependency package directory, such as the virtual environment I'm referring to here.

**Q2. Where do I put my own excel?**

If you are not familiar with the command line, you can put it in the directory where you want to generate the results, so that you can run it with a relative path, i.e. in manual. Of course, you can put it in any directory, which requires an absolute path.

Relative paths can be placed in the same directory as output when running on the command line in the local environment, just type "geochemistrypi data-mining --data Data_Regression.xlsx" to automatically access the Data_Regression data file in the same directory. Regression data files in the same directory, and the virtual environment in the same directory as C:\Users\ZW\Wen_virtual_environment. Enter the absolute path such as "geochemistrypi data-mining --data C:\Users\jmzha\Desktop\test2.xlsx".

**Q3. What graphs or models or tables can be made from the data used for the test?**

You can select the last option in the model training session, so that you can quickly generate all the model plots. However, the software is not perfect at present, so if you choose the last option, don't choose the function of AutoML first.

**Q4. Could we use a complicated absolute path of the file?**

The absolute path of any disk is fine, but the path cannot contain spaces, and if there are spaces in the file name, it should be handled with escape characters.

**Q5. Is there a specific description of each step in the test process?**

No, but the current process is a common data mining process, and we will write an abbreviated introduction afterwards.

**Q6. I'm having trouble installing our software because the download speed for Ray/Fiona is too slow or failing. How should I resolve this issue?**

To resolve the issue of slow or failed downloads for Ray/Fiona during installation, you can use the pip command with the Tsinghua mirror source, which may improve download speeds. This applies to both Mac and Windows systems. Here's the command:

```bash
pip install ray -i https://pypi.tuna.tsinghua.edu.cn/simple
```
You can refer to the video below：
+ Reference video: [Solutions to Failures in Direct pip Installation in China.mp4](https://www.bilibili.com/video/BV1zg4y1j7bx/?spm_id_from=333.999.0.0&vd_source=350db2ec0e0c3ee7f424928a21e82674).

Additionally, you can try cloning the source code from GitHub or Gitee repositories and then use the following command to install dependencies,For better download speeds, it is recommended to use the Tsinghua mirror source:

```bash
pip install -r requirements/production.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
GitHub Link: https://github.com/ZJUEarthData/geochemistrypi
Gitee Link: https://gitee.com/zju-earth-data/geochemistrypi

This approach is also suitable for developers who want to test the latest updates. For more information, refer to the "Local Deployment" section under "For Developers" in the online documentation.

+ Reference video:  [The Fastest Currently Feasible Installation Method in China—Installing from GitHub Using requirements.mp4](https://www.bilibili.com/video/BV1pM411V7iR/?spm_id_from=333.999.0.0&vd_source=350db2ec0e0c3ee7f424928a21e82674)

<br />
