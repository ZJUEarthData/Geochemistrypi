
# Add New Model To Framework

## Table of Contents

- [1. Framework - Design Pattern and Hierarchical Pipeline Architecture](#1-design-pattern)
- [2. Understand Machine Learning Algorithm](#2-understand-ml)
- [3. Construct Model Workflow Class](#3-construct-model)
  - [3.1 Add Basic Elements](#3-1-add-basic-element)
    - [3.1.1 Find File](#3-1-1-find-file)
    - [3.1.2 Define Class Attributes and Constructor](#3-1-2-define-class-attributes-and-constructors)
  - [3.2 Add Manual Hyperparameter Tuning Functionality](#3-2-add-manual)
    - [3.2.1 Define manual_hyper_parameters Method](#3-2-1-define-manaul-method)
    - [3.2.2 Create _algorithm.py File](#3-2-2-create-file)
  - [3.3 Add Automated Hyperparameter Tuning (AutoML) Functionality](#3-3-add-automl)
    - [3.3.1 Add AutoML Code to Model Workflow Class](#3-3-1-add-automl-code)
  - [3.4 Add Application Function to Model Workflow Class](#3-4-add-application-function)
    - [3.4.1 Add Common Application Functions and common_components Method](#3-4-1-add-common-function)
    - [3.4.2 Add Special Application Functions and special_components Method](#3-4-2-add-special-function)
    - [3.4.3 Add @dispatch() to Component Method](#3-4-3-add-dispatch)
  - [3.5 Storage Mechanism](#3-5-storage-mechanism)
- [4. Instantiate Model Workflow Class](#4-instantiate-model-workflow-class)
  - [4.1 Find File](#4-1-find-file)
  - [4.2 Import Module](#4-2-import-module)
  - [4.3 Define activate Method](#4-3-define-activate-method)
  - [4.4 Create Model Workflow Object](#4-4-create-model-workflow-object)
  - [4.5 Invoke Other Methods in Scikit-learn API Style](#4-5-invoke-other-methods)
  - [4.6 Add model_name to MODE_MODELS or NON_AUTOML_MODELS](#4-6-add-model-name)
- [5. Test Model Workflow Class](#5-test-model)
- [6. Completed Pull Request](#6-completed-pull-request)
- [7. Precautions](#7-precautions)


## 1. Framework - Design Pattern and Hierarchical Pipeline Architecture

Geochemistry π refers to the software design pattern "Abstract Factory", serving as the foundational framework upon which our advanced automated ML capabilities are built.

![Design Pattern](https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/aa84ab12-c95e-4282-a60e-64ba2858c437)
![Workflow Object](https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/f08885bf-1bec-4045-bf6b-82c5c18d3f8f)

The framework is a four-layer hierarchical pipeline architecture that promotes the creation of workflow obiects through a set of model selection interfaces. The critical layers of this architecture are, as follows:

1. Layer 1: the realization of ML model-associated functionalities with specific dependencies or libraries.
2. Layer 2: the abstract components of the ML model workflow class include regression, classification, clustering, and decomposition.
3. Layer 3: the scikit-learn API-style model selection interface implements the creation of ML model workflow objects.
4. Layer 4: the customized automated ML pipeline operated at the command line or through a web interface with a complete data-mining process.

<p align="center">
  <img src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/9c3ddc2b-700c-4685-b52f-f5f9a8931849" alt="Hierarchical Architecture" width="450" />
</p>

This pattern-driven architecture offers developers a standardized and intuitive way to create a ML model workflow class in Layer 2 by using a unified and consistent approach to object creation in Layer 3. Furthermore, it ensures the interchangeability of different model applications, allowing for seamless transitions between methodologies in Layer 1.

<img width="318" alt="image" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/ac2c5d7e-8eb2-4e48-811d-dc1190e50d67">

The code of each layer lies as shown above.

**Notice**: in our framework, a **model workflow class** refers to an **algorithm workflow class** and a **mode** includes multiple model workflow classes.

Now, we will take KMeans algorithm as an example to illustrate the connection between each layer. Don't get too hung up on ths part. Once you finish reading the whole article, you can come back to here again.

After reading this article, you are recommended to refer to this publication also for more details on the whole scope of our framework:

https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023GC011324


## 2. Understand Machine Learning Algorithm
You need to understand the general meaning of the machine learning algorithm you are responsible for. Then you encapsultate it as an algorithm workflow in our framework and put it under the directory `geochemistrypi/data_mining/model`. Then you need to determine which **mode** this algorithm belongs to and the role of each parameter. For example, linear regression algorithm belongs to regression mode in our framework.

+ When learning the ML algorithm, you can refer to the relevant knowledge on the [scikit-learn official website](https://scikit-learn.org/stable/index.html).


## 3. Construct Model Workflow Class

**Noted**: You can reference any existing model workflow classes in our framework to implement your own model workflow class.

### 3.1 Add Basic Elements

#### 3.1.1 Find File
First, you need to construct the algorithm workflow class in the corresponding model file. The corresponding model file locates under the path `geochemistrypi/data_mining/model`.

![image1](https://github.com/ZJUEarthData/geochemistrypi/assets/97781484/3c7d4e53-1a99-4e7e-87b6-fdcb94a9e510)

**E.g.,** If you want to add a model for the regression mode, you need to add it in the `regression.py` file.


#### 3.1.2 Define Class Attributes and Constructor

(1) Define the algorithm workflow class and its base class

```
class ModelWorkflowClassName(BaseModelWorkflowClassName):
```
+ You can refer to the ModelName of other models, the format (Upper case and with the suffix 'Corresponding Mode') needs to be consistent. E.g., `XGBoostRegression`.
+ Base class needs to be inherited according to the mode the model belongs to.

```
"""The automation workflow of using "ModelWorkflowClassName" algorithm to make insightful products."""
```
+ Class docstring, you can refer to other classes. The template is shown above.

(2) Define the class attributes `name`

```
name = "algorithm terminology"
```
+ The class attributes `name` is different from ModelWorkflowClassName. E.g., the name `XGBoost` in `XGBoostRegression` model workflow class.
+ This name needs to be added to the corresponding constant variable in `geochemistrypi/data_mining/constants.py` file and the corresponding mode processing file under the `geochemistrypi/data_mining/process` folder. Note that those name value should be identical. It will be further explained in later section.
+ For example, the name value `XGBoost` should be included in the constant varible `REGRESSION_MODELS` in `geochemistrypi/data_mining/constants.py` file and it will be use in `geochemistrypi/data_mining/process/regress.py`.

(3) Define the class attrbutes `common_functiion` or `special_function`

If this model workflow class is a base class, you need to define the class attrbutes `common_functiion`. For example, the class attrbutes `common_functiion` in the base workflow class `RegressionWorkflowBase`.

The values of `common_functiion` are the description of the functionalities of the models belonging to the same mode. It means the children class (all regession models) can share the same common functionalies as well.

```
common_functiion = []
```

If this model workflow class is a specific model workflow class, you need to define the class attrbutes `special_function`. For example, the class attrbutes `special_function` in the model workflow class `XGBoostRegression`.

The values of `special_function` are the description of the owned functionalities of that specific model. Those special functions cannot be reused by other models.

```
special_function = []
```

More detail will be explained in the later section.

(4) Define the signature of the constructor
```
def __init__(
       self,
       parameter: type = Default parameter value,
    ) -> None:
```
+ The parameters in the constructor is from the algorithm library you depend on. For example, you use **Lasso** algorithm from Sckit-learn library. You can reference its introduction ([Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)) in Scikit-learn website.
+ Default parameter value needs to be set according to scikit-learn official documents also.

![image2](https://github.com/ZJUEarthData/geochemistrypi/assets/97781484/0f02b7bb-bef1-4b56-9c84-6162e86e2093)


```
"""
Parameters
----------
parameter: type，default = Dedault

References
----------
Scikit-learn API: sklearn.model.name
https://scikit-learn.org/......
```
+ Parameters docstring are in the source code of the corresponding algorithm on the official website of sklearn.

![image3](https://github.com/ZJUEarthData/geochemistrypi/assets/97781484/0926a7e5-7243-4f4b-a3bb-bc4393b9633d)


(5) The constructor of Base class is called
```
super().__init__()
```

(6) Initializes the instance's state by assigning the parameter values passed to the constructor to the instance's attributes.
```
self.parameter=parameter
```

(7) Instantiate the algorithm class you depend on and assign. For example, `Lasso` from the library `sklearn.linear_model`.
```
self.model = modelname(
  parameter=self.parameter
)
```
**Note:** Don't forget to import the class from scikit-learn library

![image4](https://github.com/ZJUEarthData/geochemistrypi/assets/97781484/38e64144-fa19-4ef2-83d1-709504ba8001)

(8) Define the instance attribute `naming`
```
self.naming = Class.name
```
This one will be use to print the name of the class and to activate the AutoML functionality. E.g, `self.naming = LassoRegression.name`. Further explaination is in section 2.2.

(9) Define the instance attribute  `customized` and `customized_name`
```
self.customized = True
self.customized_name = "Algorithm Name"
```
These will be use to leverage the customization of AutlML functionality. E.g,`self.customized_name = "Lasso"`. Further explaination is in section 2.3.

(10) Define other instance attributes
```
self.attributes=...
```


### 3.2 Add Manual Hyperparameter Tuning Functionality

Our framework provides the user to set the algorithm hyperparameter manually or automiacally. In this part, we implement the manual functionality.

Sometimes the users want to input the hyperparameter values for model training manually, so you need to establish an interaction way to get the user's input.

#### 3.2.1 Define manual_hyper_parameters Method

The manual operation is control by the **manual_hyper_parameters** method. Inside this method, we encapsulate a lower level application function called algorithm_manual_hyper_parameters().

```
@classmethod
def manual_hyper_parameters(cls) -> Dict:
    """Manual hyper-parameters specification."""
    print(f"-*-*- {cls.name} - Hyper-parameters Specification -*-*-")
    hyper_parameters = algorithm_manual_hyper_parameters()
    clear_output()
    return hyper_parameters
```

+ The **manual_hyper_parameters** method is called in the corresponding mode operation file under the `geochemistrypi/data_mining/process` folder.

+ This lower level application function locates in the `geochemistrypi/data_mining/model/func/specific_mode` folder  which limits the hyperparameters the user can set manually. E.g., If the model workflow class `LassoRegression` belongs to the regression mode, you need to add the `_lasso_regression.py` file under the folder `geochemistrypi/data_mining/model/func/algo_regression`. Here, `_lasso_regression.py` contains all encapsulated application functions specific to lasso algorithm.
![image5](https://github.com/ZJUEarthData/geochemistrypi/assets/97781484/9d7b44d0-fd85-4a6a-a2a8-3f531475f3f6)

#### 3.2.2 Create `_algorithm.py` File

(1) Create a _algorithm.py file

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
+ You needs to choose the appropriate common utility functions according to the input type of hyperparameter.

(3) Define the application function
```
def algorithm_manual_hyper_parameters() -> Dict:
```

(4) Interactive format
```
print("Hyperparameters: Explaination")
print("A good starting value ...")
Hyperparameters = type_input(Default Value, SECTION[2], "@Hyperparameters: ")
```
**Note:** You can query ChatGPT for the recommended good starting value. The default value can come from that one in the imported library. For example, check the default value of the specific parameter for `Lasso` algorithm in [Scikit-learn Website](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html).

(5) Integrate all hyperparameters into a dictionary type and return.
```
hyper_parameters = {
    "Hyperparameters1": Hyperparameters1,
    "Hyperparameters": Hyperparameters2,
}
retuen hyper_parameters
```

#### 3.2.3 Import in The Model Workflow Class File
```
from .func.algo_mode._algorithm.py import algorithm_manual_hyper_parameters
```
![image6](https://github.com/ZJUEarthData/geochemistrypi/assets/97781484/27e74d2c-8539-41e6-bca9-f0dd50d4ed74)


### 3.3 Add Automated Hyperparameter Tuning (AutoML) Functionality

#### 3.3.1 Add AutoML Code to Model Workflow Class

Currently, only supervised learning modes (regression and classification) support AutoML. Hence, only the algorithm belonging to these two modes need to implment AutoML functionality.

Our framework leverages FLAML + Ray to build the AutoML functionality. For some algorithms, FLAML has encapsulated them. Hence, it is easy to operate with those built-in algorithm. However, some algorithms without encapsulation needs our customization on our own.

There are three cases in total:
+ C1: Encapsulated -> FLAML (Good example: `XGBoostRegression` in `regression.py`)
+ C2: Unencapsulated -> FLAML (Good example: `SVMRegression` in `regression.py`)
+ C3: Unencapsulated -> FLAML + RAY (Good example: `MLPRegression` in `regression.py`)

Here, we only talk about 2 cases, C1 and C2. C3 is a special case and it is only implemented in MLP algorithm.

Noted:

+ The calling method **fit** is defined in the base class, hence, no need to define it again in the specific model workflow class. You can refrence the **fit** method of `RegressionWorkflowBase` in `regression.py`

The following two steps is needed to implement AutoML functionality in the model workflow class. But for C1 it only requires the first step while C2 needs two step both.

(1)  Create `settings` method
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

For C1, the value of "estimator_list" should come from the specified name in [FLAML library](https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML). For example, the specified name `xgboost` in the model workflow class `XGBoostRegression`. Also we need to put this specified value inside a list.

<img width="1274" alt="image" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/3287fefa-3986-4b98-9746-fb2f348fa7e7">

For C2, the value of "estimator_list" should be the instance attribute `self.customized_name`. For example, `self.customized_name = "SVR"` in the model workflow class `SVMRegression`. Also we need to put this specified value inside a list.

**Note:** You can keep the other key-value pair consistent with other exited model workflow classes.

(2) Create `customization` method
You can add the parameter tuning code according to the following code:
```
@property
def customization(self) -> object:
    """The customized 'Your model' of FLAML framework."""
    from flaml import tune
    from flaml.data import 'TPYE'
    from flaml.model import SKLearnEstimator
    from 'sklearn' import 'model_name'

    class 'Model_Name'(SKLearnEstimator):
        def __init__(self, task=type, n_jobs=None, **config):
            super().__init__(task, **config)
            if task in 'TYPE':
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
**Note1:** The content in ' ' needs to be modified according to your specific code. You can reference that one in the model workflow class `SVMRegression`.
**Note2:**
```
space = {
    "'parameters1'": {"domain": tune.uniform(lower='...', upper='...'), "init_value": '...'},
    "'parameters2'": {"domain": tune.choice([True, False])},
    "'parameters3'": {"domain": tune.randint(lower='...', upper='...'), "init_value": '...'},
}
```

+ tune.uniform represents float
+ tune.choice represents bool
+ tune.randint represents int
+ lower represents the minimum value of the range, upper represents the maximum value of the range, and init_value represents the initial value
**Note:** You need to select parameters based on the actual situation of the model


### 3.4 Add Application Function to Model Workflow Class

We treat the insightful outputs (index, scores) or diagrams to help to analyze and understand the algorithm as useful application. For example, XGBoost algorithm can produce feature importance score, hence, drawing feature importance diagram is an **application function** we can add to the model workflow class `XGBoostRegression`.

Conduct research on the corresponding model and look for its useful application functions that need to be added.

+ You can confirm the functions that need to be added on the official website of the model (such as scikit learn), search engines (such as Google), chatGPT, etc.

In our framework, we define two types of application function: **common application function** and **special application function**.

Common application function can be shared among the model workflow classes which belong to the same mode. It will be placed inside the base model workflow class. For example, `classification_report` is a common application function placed inside the base class `ClassificationWorkflowBase`. Notice that it is encapsulated in the **private** instance method `_classification_report`.

Likewise, special application function is the special fucntionalities owned by the algorithm itself, hence it is placed inside a specific model workflow class. For example, for KMeans algorithm, we can get the inertia scores from it. Hence, inside the model workflow class `KMeansClustering`, we have a **private** instance method `_get_inertia_scores`.

Now, the next question is how to invoke these application function in our framework.

In fact, we put the invocation of the application function in the component method. Accordingly, we have two types of components:

(1) `common_components` is a public method in the base class, and all common application functions will be invoked inside.

(2) `special_components` is unique to the algorithm, so they need to be added in a specific model workflow class. All special aaplication function related to this algorithm will be invoked inside.

![Image1](https://github.com/ZJUEarthData/geochemistrypi/assets/113361635/3f983a7a-3b0d-4c7b-b7b7-31b317f4d9d0)

For more details, you can refer to the brief illustraion of the framework in section 1.

#### 3.4.1 Add Common Application Functions and `common_components` Method

`common_components` will invoke the common application functions used by all its children model workflow class, so it is necessary to consider the situation of each child model workflow class when adding a application function to it. The better way is to put the application function inside a specific child model workflow class firstly if you are not sure it can be classified as a common application function.

**1. Add common application function to the base class**

Once you’ve identified the functionality you want to add, you can define the corresponding functions in the base class.

The steps to implement are:

(1) Define the private function name and add the required parameters.
(2) Use annotations to decorate the function.
(3) Add the docstring to explain the use of this functionality.
(4) Referencing specific libraries (e.g., Scikit-learn) to implement the functionality.
(5) Change the format of data acquisition and save the produced data or images, etc.

**2. Encapsulte the concrete code in Layer 1**

Please refer to our framework's definition of **Layer 1** in section 1.

Some functions may use large code due to their complexity. To ensure the style and readability of the codebase, you need to put the specific function implementation into the corresponding `geochemistrypi/data_mining/model/func/mode/_common` files and call it.

The steps to implement are:

(1) Define the public function name, add the required parameters and proper decorator.
(2) Add the docstring to explain the use of this functionality，the significance of each parameter and the related reference.
(3) Implement functionality.
(4) Returns the value used in **Layer 2**.

**3. Define `common_components` Method**

The steps to implement are:

(1) Define the path to store the data and images, etc.
(2) Invoke the common application functions one by one.

**4. Apeend The Name of Functionality in Class Attribute `common_function`**

The steps to implement are:

(1) Create a class attribute `common_function` list in `ClusteringWorkflowBase`
(2) Create a enum class to include the name of the functionality
(3) Append the value of enum class into `common_function` list

**Example**

The following is the example of adding model evaluation score to the clustering base class.

First, you need to find the base class of clustering.

![Image2](https://github.com/ZJUEarthData/geochemistrypi/assets/113361635/b41a5af8-6cf3-4747-8c83-e613a3fee04b)

<img width="648" alt="image" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/6ca11f66-9f0f-4fbb-8c95-2ff14d0fe40d">

**1. Add `_score` function in base class `ClusteringWorkflowBase(WorkflowBase)`**

```python
@staticmethod
def _score(data: pd.DataFrame, labels: pd.DataFrame, func_name: str, algorithm_name: str, store_path: str) -> None:
    """Calculate the score of the model."""
    print(f"-----* {func_name} *-----")
    scores = score(data, labels)
    scores_str = json.dumps(scores, indent=4)
    save_text(scores_str, f"{func_name}- {algorithm_name}", store_path)
    mlflow.log_metrics(scores)
```

**2. Encapsulte the concrete code of `score` in Layer 1**

You need to add the specific function implementation `score` to the corresponding `geochemistrypi/data_mining/model/func/algo_clustering/_common` file.

![Image5](https://github.com/ZJUEarthData/geochemistrypi/assets/113361635/ee6bb43e-f30e-47b6-8d78-13f017994a44)

```python
def score(data: pd.DataFrame, labels: pd.DataFrame) -> Dict:
    """Calculate the scores of the clustering model.

    Parameters
    ----------
    data : pd.DataFrame (n_samples, n_components)
        The true values.

    labels : pd.DataFrame (n_samples, n_components)
        Labels of each point.

    Returns
    -------
    scores : dict
        The scores of the clustering model.
    """
    silhouette = silhouette_score(data, labels)
    calinski_harabaz = calinski_harabasz_score(data, labels)
    print("silhouette_score: ", silhouette)
    print("calinski_harabasz_score:", calinski_harabaz)
    scores = {
        "silhouette_score": silhouette,
        "calinski_harabasz_score": calinski_harabaz,
    }
    return scores
```

**3. Define `common_components` Method in class `ClusteringWorkflowBase(WorkflowBase)`**

```python
def common_components(self) -> None:
    """Invoke all common application functions for clustering algorithms."""
    GEOPI_OUTPUT_METRICS_PATH = os.getenv("GEOPI_OUTPUT_METRICS_PATH")
    GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
    self._score(
        data=self.X,
        labels=self.clustering_result["clustering result"],
        func_name=ClusteringCommonFunction.MODEL_SCORE.value,
        algorithm_name=self.naming,
        store_path=GEOPI_OUTPUT_METRICS_PATH,
    )
```

**4. Apeend The Name of Functionality in Class Attribute `common_function`**

Create a class attribute `common_function` in `ClusteringWorkflowBase`.

```
class ClusteringWorkflowBase(WorkflowBase):
    """The base workflow class of clustering algorithms."""

    common_function = [func.value for func in ClusteringCommonFunction]
```

The enum class should be put in the corresponding path `geochemistrypi/data-mining/model/func/algo_clustering/_enum.py`

<img width="890" alt="image" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/7a3c9e16-8d8d-4566-9516-500b3cdbfcf8">


#### 3.4.2 Add Special Application Functions and `special_components` Method

special application function is a feature that is unique to each specific model. The whole process is similar to that of previous sectoin for common functionalities.

The process is as follows:

1. Add special application function with proper decorator to the child model workflow class
2. Encapsulte the concrete code in Layer 1
3. Define `special_components` method
4. Apeend the name of functionality in class attribute `special_function`

**Example**

Each algorithms has their own characteristics. Hence, they have different special fucntionalities as well. For example, for KMeans algorithm, we can get the inertia scores from it. Hence, inside the model workflow class `KMeansClustering`, we have a **private** instance method `_get_inertia_scores`.

First, you need to find the child model workflow class for KMeans algorithm.

![Image2](https://github.com/ZJUEarthData/geochemistrypi/assets/113361635/b41a5af8-6cf3-4747-8c83-e613a3fee04b)

<img width="642" alt="image" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/53ea8041-e239-4a8f-8e83-e1c68bb90085">

**1. Add `_get_inertia_scores` function in child model workflow class `KMeansClustering(ClusteringWorkflowBase)`**

```python
@staticmethod
def _get_inertia_scores(func_name: str, algorithm_name: str, trained_model: object, store_path: str) -> None:
    """Get the scores of the clustering result."""
    print(f"-----* {func_name} *-----")
    print(f"{func_name}: ", trained_model.inertia_)
    inertia_scores = {f"{func_name}": trained_model.inertia_}
    mlflow.log_metrics(inertia_scores)
    inertia_scores_str = json.dumps(inertia_scores, indent=4)
    save_text(inertia_scores_str, f"{func_name} - {algorithm_name}", store_path)
```

**2. Encapsulte the concrete code in Layer 1**

Getting the inertia score is only one line of code, hence no need to further encapsulate it.

**3. Define `special_components` Method in class `KMeansClustering(ClusteringWorkflowBase)`**

```python
def special_components(self, **kwargs: Union[Dict, np.ndarray, int]) -> None:
    """Invoke all special application functions for this algorithms by Scikit-learn framework."""
    GEOPI_OUTPUT_METRICS_PATH = os.getenv("GEOPI_OUTPUT_METRICS_PATH")
    self._get_inertia_scores(
        func_name=KMeansSpecialFunction.INERTIA_SCORE.value,
        algorithm_name=self.naming,
        trained_model=self.model,
        store_path=GEOPI_OUTPUT_METRICS_PATH,
    )
```

![Image7](https://github.com/ZJUEarthData/geochemistrypi/assets/113361635/18dec84b-44ae-4883-a5b8-db2c6e0ef5c8)

+ Also, if only part of the models share a functionality, for example, feature importance in tree-based algorithm including XGBoost, Decision Tree, etc. Hence, you can create a Mixin class to include that application function and let the tree-based model workflow class inherit it. Such as `ExtraTreesRegression(TreeWorkflowMixin, RegressionWorkflowBase)`

**4. Apeend The Name of Functionality in Class Attribute `special_function`**

Create a class attribute `special_function` list in `KMeansClustering`.

```
class KMeansClustering(ClusteringWorkflowBase):
    """The automation workflow of using KMeans algorithm to make insightful products."""

    name = "KMeans"
    special_function = [func.value for func in KMeansSpecialFunction]
```

The enum class should be put in the corresponding path `geochemistrypi/data-mining/model/func/algo_clustering/_enum.py`

<img width="886" alt="image" src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/db69d0a7-0c3f-4943-88e3-0262ecfa47dc">


#### 3.4.3 Add `@dispatch()` to Component Method

Howerever, in **regression** mode and **classification** mode, there are two different scenarios (AutoML and manual ML) when defining either `common_components` or `special_components` method. It is needed because we need to differentiate AutoML and manual ML. For example, inside the base model workflow class `RegressionWorkflowBase`, there are two `common_components` methods but with different decorators. Also, in its child model workflow class `ExtraTreesRegression`, there are two `special_components` methods but with different decorators.

Inside our framework, we leverages the thought of **method overloading** which is not supported by Python natively but we can achieve it through a library **multipledispatch**. The invocation of `common_components` and `special_components` method locates in Layer 3 which will be explained in later section.

The differences between AutoML and manual ML are as follows:

**1. The decorator**

+ For manual ML: add @dispatch() to decorate the component method
+ For AutoML: add @dispatch(bool) to decorate the component method

**2. The signature of the component method**

For `common_compoents method`:
```
Manual ML:
@dispatch()
def common_components(self) -> None:

AutoML:
@dispatch(bool)
def common_components(self, is_automl: bool = False) -> None:
```

For `special_compoents method`:
```
Manual ML:
@dispatch()
def special_components(self, **kwargs) -> None:

AutoML:
@dispatch(bool)
def special_components(self, is_automl: bool = False, **kwargs) -> None:
```

**3. The trained model instance variable**

Usually, inside the component method, we will pass the trained model instance variable to the application function. For example, for `common_components` in `RegressionWorkflowBase(WorkflowBase)`, be careful about the value passed to the parameter `trained_model`.

```
Manual ML:
@dispatch()
def common_components(self) -> None:
  self._cross_validation(
    trained_model=self.model,
    X_train=RegressionWorkflowBase.X_train,
    y_train=RegressionWorkflowBase.y_train,
    cv_num=10,
    algorithm_name=self.naming,
    store_path=GEOPI_OUTPUT_METRICS_PATH,
  )

AutoML:
@dispatch(bool)
def common_components(self, is_automl: bool = False) -> None:
  self._cross_validation(
    trained_model=self.auto_model,
    X_train=RegressionWorkflowBase.X_train,
    y_train=RegressionWorkflowBase.y_train,
    cv_num=10,
    algorithm_name=self.naming,
    store_path=GEOPI_OUTPUT_METRICS_PATH,
)
```

**Note:** The content of this part needs to be selected according to the actual situation of your own model. Can refer to similar classes.

#### 3.5 Storage Mechanism

In Geochemistry π, the storage mechanism consists of two components: the **geopi_tracking** folder and the **geopi_output** folder. MLflow uses the geopi_tracking folder as the store for visualized operation in the web interface, which researchers cannot modify directly. The geopi_output folder is a regular folder aligning with MLflow’s storage structure, which researchers can operate. Overall, this unique storage mechanism is purpose-built to track each experiment and its corresponding runs in order to create an organized and coherent record of researchers’ scientific explorations.

<p align="center">
  <img src="https://github.com/ZJUEarthData/geochemistrypi/assets/47497750/401f3429-c44f-4b76-b085-7a9dcc987cde" alt="Storage Mechanism" width="500" />
</p>

In the codebase, we use Python's open() function to store data into the **geopi_output** folder while MLflow's methods to store data into the **geopi_tracking** folder.

The common MLflow's methods includes:

+ mlflow.log_param(): Log a parameter (e.g. model hyperparameter) under the current run.
+ mlflow.log_params(): Log a batch of params for the current run.
+ mlflow.log_metric(): Log a metric under the current run.
+ mlflow.log_metrics(): Log multiple metrics for the current run.
+ mlflow.log_artifact(): Log a local file or directory as an artifact of the currently active run. In our software, we use it to store the images, data and text.

You can refer the API document of MLflow for more details.

Actually, we have encapsulated a bunch of saving functions in `geochemistrypi/data_mining/utils/base.py`, which can be used to store the data into the **geopi_output** folder and the **geopi_tracking** folder at the same time. It includes the functions `save_fig`, `save_data`, `save_text`, `save_model`.

Usually, when you want to use the saving functions, you only need to pass it the storage path and data to store.

For example, in the case of adding a common application function into base clustering model workflow class.

![Image4](https://github.com/ZJUEarthData/geochemistrypi/assets/113361635/5e3eac82-19f8-4ef3-87a6-701ce6f9ac1b)

```
GEOPI_OUTPUT_METRICS_PATH = os.getenv("GEOPI_OUTPUT_METRICS_PATH")
```
+ This line of code gets the metrics output path from the environment variable.
```
GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_IMAGE_MODEL_OUTPUT_PATH")
```
+ This line of code gets the image model output path from the environment variable.
```
GEOPI_OUTPUT_ARTIFACTS_PATH = os.getenv("GEOPI_OUTPUT_ARTIFACTS_PATH")
```
+ This line of code takes the general output artifact path from the environment variable.

**Note:** You need to choose to add the corresponding path according to the usage in the following functions. You can look up the pre-defined pathes created inside the function `create_geopi_output_dir` in `geochemistrypi/data_mining/utils/base.py`.

**Note:** You can refer to other similar model workflow classes to complete your implementation.


## 4. Instantiate Model Workflow Class

### 4.1 Find File

Instantiating a model workflow class is the responsibilty of Layer 3. Layer 3 is represented by the scikit-learn API-style model selection interface in the corresponding mode file under the `geochemistrypi/data_mining/process` folder.

![image7](https://github.com/ZJUEarthData/geochemistrypi/assets/97781484/36e8f6ee-ae21-4f86-b000-0a373ea63cca)

**eg:** If your model workflow class belongs to regression mode, you need to implement the creation of ML model workflow objects in `regress.py` file.

### 4.2 Import Module

For example, for the model workflow class belonging to regression, you need to add your model inside `regress.py` file by using `from ..model.regression import()`.

```
from ..model.regression import(
  ...
  ModelWorkflowClass,
)
```

![image8](https://github.com/ZJUEarthData/geochemistrypi/assets/97781484/36fabb07-10b0-419a-b31d-31c036493b7b)

### 4.3 Define `activate` Method

The `activate` method defined in Layer 3 will be invoked in Layer 4.

For supervised learning (regression and classification), the signature of `activate` method is:
```
def activate(
    self,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
) -> None:
    """Train by Scikit-learn framework."""
```

For unsupervised learning (clustering, decomposition and abnormaly detection), the signature of `activate` method is:
```
def activate(
    self,
    X: pd.DataFrame,
    y: Optional[pd.DataFrame] = None,
    X_train: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.DataFrame] = None,
) -> None:
    """Train by Scikit-learn framework."""
```

The difference is that for unsupervised learning, there is no need to seperate y and split the training-testing set. But for consistency, we keep it there.

In **regression** mode and **classification** mode, there are two different scenarios (AutoML and manual ML) when defining either `activated` method. It is needed because we need to differentiate AutoML and manual ML. Hence, we still use @dispatch to decorate it. For example, in `RegressionModelSelection` class, we need to define two `activate` methods with different decorators.

```
Manual ML:
@dispatch(object, object, object, object, object, object)
def activate(
    self,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
) -> None:

AutoML:
@dispatch(object, object, object, object, object, object, bool)
def activate(
    self,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    is_automl: bool,
) -> None:
```

The differences above include the signature of @dispatch and the signature of `activate` method.

### 4.4 Create Model Workflow Object

There are two `activate` methods defined in the Regression and Classification mode, the first method uses the Scikit-learn framework, and the second method uses the FLAML and RAY frameworks. Decomposition and Clustering algorithms only use the Scikit-learn framework.  The instantiation of model workflow class inside `activate` method builds the connnectioni between Layer 3 and Layer 2.

(1) The invocatioin of model workflow class in the first activate method (Used in classification, regression,decomposition, clustering, abnormaly detection) needs to pass the hyperparameters for manual ML:
```
elif self.model_name == "ModelName":
    hyper_parameters = ModelWorkflowClass.manual_hyper_parameters()
    self.dcp_workflow = ModelWorkflowClass(
        Hyperparameters1=hyper_parameters["Hyperparameters2"],
        Hyperparameters1=hyper_parameters["Hyperparameters2"],
        ...
    )
```
+ This "ModelName" needs to be added to the corresponding constant variable in `geochemistrypi/data_mining/constants.py` file. It will be further explained in later section.

**eg:**

![image9](https://github.com/ZJUEarthData/geochemistrypi/assets/97781484/d4d3c208-e7a5-4e5c-a403-1fa6646bf7a7)

（2）The invocatioin of model workflow class in the second activate method（Used in classification, regression）for AutoML:
```
elif self.model_name == "ModelName":
  self.reg_workflow = ModelWorkflowClass()
```
+ This "ModelName" needs to be added to the corresponding constant variable in `geochemistrypi/data_mining/constants.py` file. It will be further explained in later section.

**eg:**
![image10](https://github.com/ZJUEarthData/geochemistrypi/assets/97781484/0eae64d1-8e50-4a02-bf08-c9fc543130d0)

### 4.5 Invoke Other Methods in Scikit-learn API Style

It should contain at least these functoins below:

+ data_upload(): Load the required data into the base class's attributes.
+ show_info(): Display what application functions the algorithm will provide.
+ fit(): Fit the model.
+ save_hyper_parameters(): Save the model hyper-parameters into the storage.
+ common_components(): Invoke all common application functions.
+ special_components(): Invoke all special application functions.
+ model_save(): Save the trained model.

You can refer to other existing mode inside `geochemistrypi/data_mining/process/mode.py` to see what other else you need.

### 4.6 Add `model_name` to `MODE_MODELS` or `NON_AUTOML_MODELS`

Find the `constants.py` file under `geochemistrypi/data_mining` folder to add the model name which should be identical to that in `geochemistrypi/data_mining/process/mode.py` and in `geochemistrypi/data_mining/model/mode.py`.

![image11](https://github.com/ZJUEarthData/geochemistrypi/assets/97781484/84544ad9-44aa-4fb4-b0f1-668f4c3da65f)

**(1) Add `model_name` to `MODE_MODELS`**

Append `model_name` to the `MODE_MODELS` list corresponding to the mode in the constants file.

**eg:** Add the name of the Lasso regression algorithm to `REGRESSION_MODELS` list.

![image12](https://github.com/ZJUEarthData/geochemistrypi/assets/97781484/ec647037-2467-4a86-b7bb-e009a48cb964)

**(2) Add `model_name` to `NON_AUTOML_MODELS`**

Only for those algorithms, they belong to either regression or classification and don't need to provide AutoML functionality. They need to append `model_name` to `NON_AUTOML_MODELS` list.

**eg:** Add the name of the Linear Regression algorithm to `NON_AUTOML_MODELS` list.

![image13](https://github.com/ZJUEarthData/geochemistrypi/assets/97781484/d6b03566-a833-4868-8738-be09d7356c9c)


## 5. Test Model Workflow Class

After the model workflow class is added, you can test it through running the command `python start_cli_pipeline.py` on the terminal. If the test reports an error, you need to debug and fix it. If there is no error, it can be submitted.


## 6. Completed Pull Request

After the test is correct, you can complete the pull request according to the puu document instructions in [Geochemistry π - Completed Pull Request](https://geochemistrypi.readthedocs.io/en/latest/index.html)

![image](https://github.com/ZJUEarthData/geochemistrypi/assets/97781484/e95c2e44-21f7-44af-8e32-e857189a5204)


## 7. Precautions
**Note1:** This tutorial only discusses the general process of adding a model, and the specific addition needs to be combined with the actual situation of the model to accurately add relevant codes.
**Note2:** If there are unclear situations and problems during the adding process, communicate with other people in time to solve them.
