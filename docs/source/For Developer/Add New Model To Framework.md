
# Add New Model To Framework




## Table of Contents

- [1. Understand the model](#1-understand-the-model)
- [2. Add Model](#2-add-model)
  - [2.1 Add The Model Class](#21-add-the-model-class)
    - [2.1.1 Find Add File](#211-find-add-file)
    - [2.1.2 Define class properties and constructors, etc.](#212-define-class-properties-and-constructors-etc)
    - [2.1.3 Define manual\_hyper\_parameters](#213-define-manual_hyper_parameters)
    - [2.1.4 Define special\_components](#214-define-special_components)
  - [2.2 Add AutoML](#22-add-automl)
    - [2.2.1 Add AutoML code to class](#221-add-automl-code-to-class)
  - [2.3 Get the hyperparameter value through interactive methods](#23-get-the-hyperparameter-value-through-interactive-methods)
    - [2.3.1 Find file](#231-find-file)
    - [2.3.2 Create the .py file and add content](#232-create-the-py-file-and-add-content)
    - [2.3.3 Import in the file that defines the model class](#233-import-in-the-file-that-defines-the-model-class)
  - [2.4 Call Model](#24-call-model)
    - [2.4.1 Find file](#241-find-file)
    - [2.4.2 Import module](#242-import-module)
    - [2.4.3 Call model](#243-call-model)
  - [2.5 Add the algorithm list and set NON\_AUTOML\_MODELS](#25-add-the-algorithm-list-and-set-non_automl_models)
    - [2.5.1 Find file](#251-find-file)
- [3. Test model](#3-test-model)
- [4. Completed Pull Request](#4-completed-pull-request)
- [5. Precautions](#5-precautions)


## 1. Understand the model
You need to understand the general meaning of the model, determine which algorithm the model belongs to and the role of each parameter.
+ You can choose to learn about the relevant knowledge on the [scikit-learn official website](https://scikit-learn.org/stable/index.html).


## 2. Add Model
### 2.1 Add The Model Class
#### 2.1.1 Find Add File
First, you need to define the model class that you need to complete in the corresponding algorithm file. The corresponding algorithm file is in the `model` folder in the `data_mining` folder in the `geochemistrypi` folder.

![image1](https://github.com/ZJUEarthData/geochemistrypi/assets/97781484/3c7d4e53-1a99-4e7e-87b6-fdcb94a9e510)

**eg:** If you want to add a model to the regression algorithm, you need to add it in the `regression.py` file.


#### 2.1.2 Define class properties and constructors, etc.
(1) Define the class and the required Base class
```
class NAME (Base class):
```
+ NAME is the name of the algorithm, you can refer to the NAME of other models, the format needs to be consistent.
+ Base class needs to be selected according to the actual algorithm requirements.

```
"""The automation workflow of using "Name" to make insightful products."""
```
+ Class explanation, you can refer to other classes.

(2) Define the name and the special_function

```
name = "name"
```
+ Define name, different from NMAE.
+ This name needs to be added to the _`constants.py`_ file and the corresponding algorithm file in the `process` folder. Note that the names are consistent.
```
special_function = []
```
+ special_function is added according to the specific situation of the model, you can refer to other similar models.

(3) Define constructor
```
def __init__(
       self,
       parameter:type=Default parameter value,
    ) -> None:
```
+ All parameters in the corresponding model function need to be written out.
+ Default parameter value needs to be set according to official documents.

```
 """
Parameters
----------
parameter:type，default = Dedault

References
----------
Scikit-learn API: sklearn.model.name
https://scikit-learn.org/......
```
+ Parameters is in the source of the corresponding model on the official website of sklearn

**eg:** Take the [Lasso algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) as a column.

![image2](https://github.com/ZJUEarthData/geochemistrypi/assets/97781484/0f02b7bb-bef1-4b56-9c84-6162e86e2093)

![image3](https://github.com/ZJUEarthData/geochemistrypi/assets/97781484/0926a7e5-7243-4f4b-a3bb-bc4393b9633d)

+ References is your model's official website.

(4) The constructor of Base class is called
```
super().__init__()
```
(5) Initializes the instance's state by assigning the parameter values passed to the constructor to the instance's properties.
```
self.parameter=parameter
```
(6) Create the model and assign
```
self.model=modelname(
  parameter=self.parameter
)
```
**Note:** Don't forget to import Model from scikit-learn

![image4](https://github.com/ZJUEarthData/geochemistrypi/assets/97781484/38e64144-fa19-4ef2-83d1-709504ba8001)

(7) Define other class properties
```
self.properties=...
```

#### 2.1.3 Define manual_hyper_parameters
manual_hyper_parameters gets the hyperparameter value by calling the manual hyperparameter function, and returns hyper_parameters.
```
hyper_parameters = name_manual_hyper_parameters()
```
+ This function calls the corresponding function in the `func` folder (needs to be written, see 2.2.2) to get the hyperparameter value.

+ This function is called in the corresponding file of the `Process` folder (need to be written, see 2.3).
+ Can be written with reference to similar classes


#### 2.1.4 Define special_components
Its purpose is to Invoke all special application functions for this algorithms by Scikit-learn framework.
**Note:** The content of this part needs to be selected according to the actual situation of your own model.Can refer to similar classes.

```
GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
```
+ This line of code gets the image model output path from the environment variable.
```
GEOPI_OUTPUT_ARTIFACTS_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_PATH")
```
+ This line of code takes the general output artifact path from the environment variable.
**Note:** You need to choose to add the corresponding path according to the usage in the following functions.

### 2.2 Add AutoML
#### 2.2.1 Add AutoML code to class
(1) Set AutoML related parameters
```
    @property
    def settings(self) -> Dict:
        """The configuration of your model to implement AutoML by FLAML framework."""
        configuration = {
            "time_budget": '...'
            "metric": '...',
            "estimator_list": '...'
            "task": '...'
        }
        return configuration
```
+ "time_budget" represents total running time in seconds
+ "metric" represents Running metric
+ "estimator_list" represents list of ML learners
+ "task" represents task type
**Note:** You can keep the parameters consistent, or you can modify them to make the AutoML better.

(2) Add parameters that need to be AutoML
You can add the parameter tuning code according to the following code:
```
    @property
    def customization(self) -> object:
        """The customized 'Your model' of FLAML framework."""
        from flaml import tune
        from flaml.data import 'TPYE'
        from flaml.model import SKLearnEstimator
        from sklearn.ensemble import 'model_name'

        class 'Model_Name'(SKLearnEstimator):
            def __init__(self, task=type, n_jobs=None, **config):
                super().__init__(task, **config)
                if task in 'TOYE':
                    self.estimator_class = 'model_name'

            @classmethod
            def search_space(cls, data_size, task):
                space = {
                    "'parameters1'": {"domain": tune.uniform(lower='...', upper='...'), "init_value": '...'},
                    "'parameters2'": {"domain": tune.choice([True, False])},
                    "'parameters3'": {"domain": tune.randint(lower='...', upper='...'), "init_value": '...'},
                }
                return space

        return "Model_Name"
```
**Note1:** The content in ' ' needs to be modified according to your specific code
**Note2:**
```
                space = {
                    "'parameters1'": {"domain": tune.uniform(lower='...', upper='...'), "init_value": '...'},
                    "'parameters2'": {"domain": tune.choice([True, False])},
                    "'parameters3'": {"domain": tune.randint(lower='...', upper='...'), "init_value": '...'},
                }
```

+ tune.Uniform represents float
+ tune.choice represents bool
+ tune.randint represents int
+ lower represents the minimum value of the range, upper represents the maximum value of the range, and init_value represents the initial value
**Note:** You need to select parameters based on the actual situation of the model

(3) Define special_components(FLAML)
This part is the same as 2.1.4 as a whole, and can be modified with reference to it, but only the following two points need to be noted:
a.The multi-dispatch function is different
Scikit-learn framework：@dispatch()
FLAML framework：@dispatch(bool)

b.Added 'is_automl: bool' to the def
**eg:**
```
Scikit-learn framework：
def special_components(self, **kwargs) -> None:

FLAML framework：
def special_components(self, is_automl: bool, **kwargs) -> None:
```
c.self.model has a different name
**eg:**
```
Scikit-learn framework：
coefficient=self.model.coefficient

FLAML framework：
coefficient=self.auto_model.coefficient
```

**Note:** You can refer to other similar codes to complete your code.

### 2.3 Get the hyperparameter value through interactive methods
Sometimes the user wants to modify the hyperparameter values for model training, so you need to establish an interaction to get the user's modifications.


#### 2.3.1 Find file
You need to find the corresponding folder for model. The corresponding algorithm file is in the `func` folder in the model folder in the `data_mining` folder in the `geochemistrypi` folder.

![image5](https://github.com/ZJUEarthData/geochemistrypi/assets/97781484/9d7b44d0-fd85-4a6a-a2a8-3f531475f3f6)

**eg:** If your model belongs to the regression, you need to add the `corresponding.py` file in the `alog_regression` folder.


#### 2.3.2 Create the .py file and add content
(1) Create a .py file
**Note:** Keep name format consistent.

(2) Import module
```
from typing import Dict
from rich import print
from ....constants import SECTION
```
+ In general, these modules need to be imported
```
from ....data.data_readiness import bool_input, float_input, num_input
```
+ This needs to choose the appropriate import according to the hyperparameter type of model interaction.

(3) Define the function
```
def name_manual_hyper_parameters() -> Dict:
```
**Note:** The name needs to be consistent with that in 2.1.3.

(4) Interactive format
```
print("Hyperparameters: Role")
print("Recommended value")
Hyperparameters = type_input(Recommended value, SECTION[2], "@Hyperparameters: ")
```
**Note:** The recommended value needs to be the default value of the corresponding package.

(5) Integrate all hyperparameters into a dictionary type and return.
```
hyper_parameters = {
        "Hyperparameters1": Hyperparameters1,
        "Hyperparameters": Hyperparameters2,
}
retuen hyper_parameters
```

#### 2.3.3 Import in the file that defines the model class
```
from .func.algo_regression.Name import name_manual_hyper_parameters
```
**eg:**
![image6](https://github.com/ZJUEarthData/geochemistrypi/assets/97781484/27e74d2c-8539-41e6-bca9-f0dd50d4ed74)

### 2.4 Call Model

#### 2.4.1 Find file
Call the model in the corresponding file in the `process` folder. The corresponding algorithm file is in the `process` folder in the` model` folder in the `data_mining` folder in the `geochemistrypi` folder.

![image7](https://github.com/ZJUEarthData/geochemistrypi/assets/97781484/36e8f6ee-ae21-4f86-b000-0a373ea63cca)

**eg:** If your model belongs to the regression,you need to call it in the regress.py file.

#### 2.4.2 Import module
You need to add your model in the from ..model.regression import().
```
from ..model.regression import(
  ...
  NAME,
)
```
**Note:** NAME needs to be the same as the NAME when defining the class in step 2.1.2.
**eg:**

![image8](https://github.com/ZJUEarthData/geochemistrypi/assets/97781484/36fabb07-10b0-419a-b31d-31c036493b7b)

#### 2.4.3 Call model
There are two activate methods defined in the Regression and Classification algorithms, the first method uses the Scikit-learn framework, and the second method uses the FLAML and RAY frameworks. Decomposition and Clustering algorithms only use the Scikit-learn framework. Therefore, in the call, Regression and Classification need to add related codes to implement the call in both methods, and only one time is needed in Clustering and Decomposition.

(1) Call model in the first activate method(Including Classification, Regression,Decomposition,Clustering)
```
elif self.model_name == "name":
            hyper_parameters = NAME.manual_hyper_parameters()
            self.dcp_workflow = NAME(
                Hyperparameters1=hyper_parameters["Hyperparameters2"],
                Hyperparameters1=hyper_parameters["Hyperparameters2"],
                ...
            )
```
+ The name needs to be the same as the name in 2.4
+ The hyperparameters in NAME() are the hyperparameters obtained interactively in 2.2
**eg:**
![image9](https://github.com/ZJUEarthData/geochemistrypi/assets/97781484/d4d3c208-e7a5-4e5c-a403-1fa6646bf7a7)

（2）Call model in the second activate method（Including Classification, Regression）
```
elif self.model_name == "name":
  self.reg_workflow = NAME()
```
+ The name needs to be the same as the name in 2.4
**eg:**
![image10](https://github.com/ZJUEarthData/geochemistrypi/assets/97781484/0eae64d1-8e50-4a02-bf08-c9fc543130d0)

### 2.5 Add the algorithm list and set NON_AUTOML_MODELS

#### 2.5.1 Find file
Find the constants file to add the model name,The constants file is in the `data_mining` folder in the `geochemistrypi` folder.

![image11](https://github.com/ZJUEarthData/geochemistrypi/assets/97781484/84544ad9-44aa-4fb4-b0f1-668f4c3da65f)

(1) Add the model name
Add model name to the algorithm list corresponding to the model in the constants file.
**eg:** Add the name of the Lasso regression algorithm.
![image12](https://github.com/ZJUEarthData/geochemistrypi/assets/97781484/ec647037-2467-4a86-b7bb-e009a48cb964)

（2）set NON_AUTOML_MODELS
Because this is a tutorial without automatic parameters, you need to add the model name in the NON_AUTOML_MODELS.
**eg:**
![image13](https://github.com/ZJUEarthData/geochemistrypi/assets/97781484/d6b03566-a833-4868-8738-be09d7356c9c)

## 3. Test model
After the model is added, it can be tested. If the test reports an error, it needs to be checked. If there is no error, it can be submitted.

## 4. Completed Pull Request
After the model test is correct, you can complete the pull request according to the puu document instructions in [Geochemistry π](https://geochemistrypi.readthedocs.io/en/latest/index.html)
![image](https://github.com/ZJUEarthData/geochemistrypi/assets/97781484/e95c2e44-21f7-44af-8e32-e857189a5204)

## 5. Precautions
**Note1:** This tutorial only discusses the general process of adding a model, and the specific addition needs to be combined with the actual situation of the model to accurately add relevant codes.
**Note2:** If there are unclear situations and problems during the adding process, communicate with other people in time to solve them.
