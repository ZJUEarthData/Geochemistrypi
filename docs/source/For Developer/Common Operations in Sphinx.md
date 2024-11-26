# Common Operations in Sphinx

Takeaway
+ Installation - PyPi(windpows, CMD)
+ Installation - PyPi(windpows, Powershell)
+ Installation - Conda(windpows, waiting for update)
+ Installation - PyPi(MacOS/Linux, waiting for update)
+ Installation - Conda(MacOS/Linux, waiting for update)
+ Fast Build Sphinx

1. Open the **CMD** terminal in the **Windows** and install the **Sphinx** package by the **Pypi**. The first step is to enter the **geochemistrypi** file and activate the virtual envirnment. Then use the **pip** command to install the **Sphinx** package.

```None
cd geochemistry
conda activate your_env # Conda with Vscode is the most recommended way
pip install -U sphinx
```
 <img width="920" alt="image" src="https://user-images.githubusercontent.com/66153455/265470689-396caf7f-4a7d-4776-aedf-b86bc6208549.jpg">

 <img width="920" alt="image" src="https://user-images.githubusercontent.com/66153455/265470690-ae786ecf-f39e-4993-8ae2-409b388282b0.jpg">

2. Open the **Powershell** terminal in the **Windows** and install the **Sphinx** package by the **Pypi**. The steps are close to the CMD way.

<img width="920" alt="image" src="https://user-images.githubusercontent.com/66153455/265470688-9a4f24ea-ce70-4608-ac70-c270f330e14f.jpg">

 <img width="920" alt="image" src="https://user-images.githubusercontent.com/66153455/265470678-21ae50d6-6934-4adc-bd42-7952cd3f661a.jpg">

3. The ***Sphinx*** ia a convenient tool for documentations in the projects managemnet. From the *[Getting start](https://www.sphinx-doc.org/en/master/tutorial/getting-started.html)*  of the official ***[Sphinx documentations](https://www.sphinx-doc.org/en/master/index.html)***, we'd better to set a *README.rst* file in a project like:

<center class="half">
    <img width="150" height="200" alt="image" src="https://user-images.githubusercontent.com/66153455/265570728-08acdf8d-0e7f-4935-a9e1-05693035c1e0.jpg"><img width="500" height="200" alt="image" src="https://user-images.githubusercontent.com/66153455/265570736-24ed3ab1-6b49-4677-847e-0a55fe2e548f.jpg">
</center>

This time I just make a trail docs file. In fact, in a real project, we must write down the descriptions of the project.

In addition, the *[Creating the documentation layout](https://www.sphinx-doc.org/en/master/tutorial/getting-started.html)* is also useful to set up a layout. The specific code is just:

```None
sphinx-quickstart docs
```

*In this doc, the consequent screenshots are gained on the Powershell/Windows platform. The fast layout would be like:*

<img width="920" alt="image" src="https://user-images.githubusercontent.com/66153455/265570738-a464efef-5187-499b-8c31-9dfcdac43104.jpg">

As you can see, after you write down the commands, some Chinese reflects are shown in the screen. The root can be set as you wish.

*Then several parameters can be set in the Terminal:*

<img width="920" alt="image" src="https://user-images.githubusercontent.com/66153455/265570740-c6cf9f23-1a53-47fa-8503-cf7c51f403a3.jpg">

*Congratulations！Now you can start write in the index.rst file:*

<img width="920" alt="image" src="https://user-images.githubusercontent.com/66153455/265577459-cdc9cb67-92e4-4ed5-a161-b04f8d988438.jpg">
