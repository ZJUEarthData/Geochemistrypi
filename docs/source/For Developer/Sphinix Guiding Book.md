## Brief Introduction of Sphinx

Sphinx is written in Python and supports Python 3.9+. It builds upon the shoulders of many third-party libraries such as Docutils and Jinja, which are installed when Sphinx is installed.

Sphinx is an amazing tool for writing beautiful documentation. Originally created for the documentation of the Python programming language, it is now popular across numerous developer communities. Most major Python libraries are documented with Sphinx, e.g. NumPy, SciPy, Scikit-Learn, Matplotlib, Django.

## Install by PyPi

For Mac, Window and Linux systems, the most convenient way to install the ***Sphinx*** package is the **Pypi**. The first step is to enter the **geochemistrypi** file and activate the virtual envirnment by:

```None
cd geochemistry
conda activate your_env # Conda with Vscode is the most recommended way
```

*In Windows/CMD, the operation is:*

 <img width="920" alt="image" src="https://user-images.githubusercontent.com/66153455/265470689-396caf7f-4a7d-4776-aedf-b86bc6208549.jpg">

*In Windows/Powershell, the operation is:*

<img width="920" alt="image" src="https://user-images.githubusercontent.com/66153455/265470688-9a4f24ea-ce70-4608-ac70-c270f330e14f.jpg">

After these preparations, we just need to use ***pip*** to finish the installation via:

```None
pip install -U sphinx
```

*In Windows/CMD, the Installation operation is:*

 <img width="920" alt="image" src="https://user-images.githubusercontent.com/66153455/265470690-ae786ecf-f39e-4993-8ae2-409b388282b0.jpg">

 *In Windows/Powershell, the Installation operation is:*

 <img width="920" alt="image" src="https://user-images.githubusercontent.com/66153455/265470678-21ae50d6-6934-4adc-bd42-7952cd3f661a.jpg">

 ## Fast start of the ***Sphinx***

 The ***Sphinx*** ia a convenient tool for documentations in the projects managemnet. From the *[Getting start](https://www.sphinx-doc.org/en/master/tutorial/getting-started.html)*  of the official ***[Sphinx documentations](https://www.sphinx-doc.org/en/master/index.html)***, we'd better to set a *README.rst* file in a project like:

<center class="half">
    <img width="165" height="200" alt="image" src="https://user-images.githubusercontent.com/66153455/265570728-08acdf8d-0e7f-4935-a9e1-05693035c1e0.jpg"><img width="550" height="200" alt="image" src="https://user-images.githubusercontent.com/66153455/265570736-24ed3ab1-6b49-4677-847e-0a55fe2e548f.jpg">
</center>

This time I just make a trail docs file. In fact, in a real project, we must write down the descriptions of the project.

In addition, the *[Creating the documentation layout](https://www.sphinx-doc.org/en/master/tutorial/getting-started.html)* is also useful to set up a layout. The specific code is just:

```None
sphinx-quickstart docs
```

*In this doc, the consequent screenshots are gained on the Poweshell/Windows platform. The fast layout would be like:*

<img width="920" alt="image" src="https://user-images.githubusercontent.com/66153455/265570738-a464efef-5187-499b-8c31-9dfcdac43104.jpg">

As you can see, after you write down the commands, some Chinese reflects are shown in the screen. The root can be set as you wish.

*Then several parameters can be set in the Terminal:*

<img width="920" alt="image" src="https://user-images.githubusercontent.com/66153455/265570740-c6cf9f23-1a53-47fa-8503-cf7c51f403a3.jpg">

*Congratulations！Now you can start write in the index.rst file:*

<img width="920" alt="image" src="https://user-images.githubusercontent.com/66153455/265577459-cdc9cb67-92e4-4ed5-a161-b04f8d988438.jpg">

------------------------------------------------------------------------------------
Summary log@2023/09-05@KeranLi: In this cycle, I started to use ***Sphinx*** to manage documentations. I just learned how to install and quick-start. So many details are still waiting for exploring.
