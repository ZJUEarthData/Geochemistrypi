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

<br />
