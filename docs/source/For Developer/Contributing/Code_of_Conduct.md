#  Code of Conduct

Contents：

1. Naming Conventions
2. Space Usage
3. Variable Type Hint
4. TODO or FIXME Comments
5. Docstring
6. Test Module
7. Git Commit Message

P.S.：Your suggestions are welcome if there is anything which you think needs to be added during the development process.👏


##  1. Naming Conventions

+ Function：Underscore Case, in which using an underscore(_) to separate one word from the next，and the words are generally lowercase.


```python
def foo_name(param):
    pass
```

+ Class：Upper Camel Case


```python
class FooName():
    pass
```

+ Variable：Underscore Case, in which using an underscore(_) to separate one word from the next，and the words are generally lowercase.


```python
var_name = 3
```

+ Global Variables：Using an underscore(_) to separate the uppercase letter.


```python
# We usually do not use it in a single.py file, but define it at the top level of global_variable.py.
CONT_VAR = 'Geochemistry Py'
```

## 2. Space Usage

+ Equal sign '=' : The use of Spaces around the equals sign for variable assignment and function parameter assignment is different.


```python
# When we assign a value to a variable, there is a space before and after the '='.
var = 1

# When function arguments are called and assigned, Spaces are not required before and after '='.
def foo(param=1):
    pass

var = method(param='yes')

# When assigning a default value to a type hint, Spaces are required before and after '='. For what is a type hint, please read the second half of this documentation.
def foo(param: int = 1) -> None:
		pass
```


+ Binary operator: There are two cases, whether it is used in the position of the argument. And we have to consider the operator precedence.


```python
# When we use a binary operator in the function argument position, we do not type a Space before or after it.
var = method(param+1)

# When we use it outside the parameter position, we need to type a Space both before and after it.
var = 3 + 2
var += 2

# We only need to add Spaces around the operators with the lowest priority, and nothing else.
var = 3*2 + 1
var = (3+2) * (2+1)
```


+ Annotation: Use **#** with a space followed by a capital letter.


```python
# It prints a word
print('Geochemistry Py')
```

+ Refer to the English format and leave a space after each punctuation mark.


```python
var1, var2 = 1, 2
var3 = method(1, 3, 2)

def foo(name=1, signal='yes'):
    pass
```

##  3. Variable Type Hint

+ Type hint: It is a type hint and check for IDEs (e.g. Pycharm) parameters, which can specify the type of the parameter and the return type of the function.


```python
# What is the type hint and what does it do specifically refer to this link ：https://sikasjc.github.io/2018/07/14/type-hint-in-python/

# Below is a list of the types that our members might be involved in, along with examples.

# str，int，return
# Function foo takes argument p1 of type str, argument p2 of type int, and returns type str.
def foo(p1: str, p2: int) -> str:
    return 'hi'


# list，tuple, set
# Function foo takes argument p1 of type list, and each element in the list is of type int.
# Parameter p2 is passed in of type tuple, and each element in the tuple is of type int.
# Parameter p3 is passed in of type set, and each element in the collection is of type str.
# Function has no return value.
from typing import List, Tuple, Set
def foo(p1: List[int], p2: Tuple[int], p3: Set[str]) -> None:
  	print(p1)
    print(p2)
    print(p3)


# dict, nested
# Function foo takes argument p1 of type dict, where the key is of type str and the value of type int.
# Parameter p2 is passed as a list of type, and each element in the list is of type tuple.
from typing import Dict, List, Tuple
def foo(p1: Dict[str: int], p2: List[Tuple[str]]) -> None:
  	print(p1)
    print(p2)


# bool，default value, any type
# Function foo takes parameter p1 as bool, which defaults to True, and parameter p2 can be of any type.
from typing import Any
def foo(p1: bool = True, p2: Any) -> None:
  	if p1:
      	print(p2)


# Union: we use it when there is a parameter with multiple possible data types.
# The p1 argument to foo can be of type list or str.
from typing import Union, List
def foo(p1: Union[List, str]) -> None:
		pass


# Optional is a simplification of Union. The data type of this parameter can be str in addition to None.
# So Optional[str] is equivalent to Union[str, None], where None is different from the function with the default argument None, which cannot be ignored.
from typing import Optional
def foo(p1: Optional[str] = None) -> str:
  	return 'Geochemistry Py'


# Tuple: used when returning multiple values.
# Function foo return two values whose types are str and list respectively
from typing import Tuple, List
def foo() -> Tuple[str, List[int]]:
    a = 'Geochemistry Py'
    b = [1, 2, 3]
    return a, b


# pandas.DataFrame, pandas.Series, numpy.ndarray
# The input type of parameter p1 is pandas.DataFrame and type of parameter p2 is pandas.Series,
# The input type of parameter p3 is pandas.Index and type of parameter p4 is numpy.ndarray.
import pandas as pd
import numpy as np
def f2(p1: pd.DataFrame, p2: pd.Series, p3: pd.Index, p4: np.ndarray) -> None:
    print(p1)
    print(p2)
```



## 4. TODO or FIXME Comments

+ 'TODO' or 'FIXME' should be followed by your name and email to inform others of what to add or change in this code.

+ We can quickly retrieve the number and location of TODOs using an IDE (e.g. Pycharm).

  ```python
  # TODO(Sany hecan@mail2.sysu.edu.cn): Append a new function for ...
  # FIXME(Sany hecan@mail2.sysu.edu.cn): Correct this error of ...
  ```



## 5. Docstring

+ Docstring is a string that appears on the first line of a module, function, class or method definition, and this docstring is used as the \__doc__ property of that object;
+ From a specification point of view, all modules should have docstrings, including the functions and classes introduced from the module;
+ All content types of Docstring are as follows, and not every docstring has to contain all the following content types, just the ones we need.


```python
"""Briefly describe the function the code is intended to achieve.

Parameters
----------
We can explain the data type, data meaning, and source of the parameters defined by the function.
parameter name : data type
    (four blanks) Parameter Meaning / Source

temperature : int, str
    Temperature at which minerals are formed. From ‘temperature.xlsx’.

density : float
		Density of the minerals.

Returns
-------
We can interpret the type and meaning of the data in the output content of the function.
parameter name : data type
    (four blanks) Parameter Meaning / Source

temperature : int, str
    Temperature at which minerals are formed. From ‘temperature.xlsx’.

References
----------
The source of the code citation can be a literature, website, book, and other sources. When citing multiple documents, you need to indicate [1][2][3] with a serial number and start a separate line.
Format of journal/book : Author + year of publication + title of article + title of journal + volume number + page number
Format of the page/website : author + year and month of publication + title + website name + website link
Note: When the number of authors is greater than two, it can be written as 'first author's name + et al.', e.g. Digne M., et al.
[1] referecne1
[2] reference2

Todo
----
To-do list. When there is more than one to-do list, we need to mark (1)(2)(3) with a serial number and start a new line.
(1) Task1
(2) Task2

Notes
-----
This section includes the requirements for the data, a description of the relevant operations, problems and solutions that may be encountered, and background on the mathematical formulae (using Latex syntax).
When the Note section contains more than one note, it needs to be marked with a serial number (1)(2)(3) or the * symbol, and on a separate line.
(1) Math :
    		sin(2k\pi + \alpha)= sin\alpha·cos(2k\pi + \alpha)= cos\alpha·tan(2k\pi + \alpha)
(2) Data :
        The data should from  ‘temperature.xlsx’.
(3) Problem :
   			When you run,you may meet the problem like...

Examples
--------
Sample doctest tests. Please refer to the Test Module of 《Geochemistry Py - Code Specification》 document for details.
>>> # comments
>>> foo(arg1, arg2)
return value

See also
--------
It identifies the location of other related codes for easy viewing.
When there are multiple related codes, they need to be marked with serial numbers (1)(2)(3) or * symbols, and marked on a separate line.
(1) Code : ’pyrolite.geochem.ind_first’
(2) Code : ’pyrolite.geochem.ind_second’
"""
```





+ Samples


```python
def foo(arg1: int, arg2: int = 1) -> int:
    """The function for what ...

    Parameters
		----------
		arg1 : int
				The actual meaning of arg1 ...
		arg2 : int, default=1
				The actual meaning of arg2 ...

		Returns
		-------
		value : int
				The actual meaning of value ...
    """
    value = arg1 + arg2
    return value
```



## 6. Test Module

+ After finishing the code packaging work, we need to test the code.
+ One way to develop high quality software is to develop test code for each feature and present test samples in both doctest and unittest formats.
+ The test code needs to provide at least 2 or more test cases and try to take into account the boundary conditions of the code.
+ The reason why unit testing is needed, common unit testing approaches and their linkages (Optional), please refer to following links：
  + https://python-course.eu/advanced-python/tests-doctest-unittest.php
  + https://python-course.eu/advanced-python/pytest.php

### (1) Doctest

+ Doctest: Add test code to the **Examples** section of the Docstring of the necessary functions, methods and classes.
+ It enhances the documentation with examples that can be used to confirm that the results of the code are consistent with the documentation.
+ Please refer to the link to use Doctest：https://docs.python.org/3/library/doctest.html
+ Here is the passed example and the results displayed on the command line.


```python
# The passed example

def foo(arg1: int, arg2: int = 1) -> int:
    """The function for what ...

    Parameters
		----------
		arg1 : int
				The actual meaning of arg1 ...
		arg2 : int, default=1
				The actual meaning of arg2 ...

		Returns
		-------
		value : int
				The actual meaning of value ...

		Examples
		--------
		>>> # run the python command
		>>> foo(2, 7)
		9
		>>> foo(1, 2)
		3
    """
    value = arg1 + arg2
    return value

if __name__ == "__main__":
  	# test whether doctest works well, notice that don't cover this part when submitting your code on GitHub
		import doctest
    doctest.testmod()
```


```python
$ python3 test.py -v
Trying:
    foo(2, 7)
Expecting:
    9
ok
Trying:
    foo(1, 2)
Expecting:
    3
ok
1 items had no tests:
    __main__
1 items passed all tests:
   2 tests in __main__.foo
2 tests in 2 items.
2 passed and 0 failed.
Test passed.
```

+ Here is the  failed example and the results displayed on the command line.


```python
# The failed example

def foo(arg1: int, arg2: int = 1) -> int:
    """The function for what ...

    Parameters
		----------
		arg1 : int
				The actual meaning of arg1 ...
		arg2 : int, default=1
				The actual meaning of arg2 ...

		Returns
		-------
		value : int
				The actual meaning of value ...

		Examples
		--------
		>>> # run the python command
		>>> foo(2, 7)
		7
		>>> foo(1, 2)
		3
    """
    value = arg1 + arg2
    return value

if __name__ == "__main__":
  	# see whether doctest works well, notice that don't cover this part when submitting your code on GitHub
		import doctest
    doctest.testmod()
```


```python
$ python3 test.py -v
Trying:
    foo(2, 7)
Expecting:
    7
**********************************************************************
File "/Users/can/Documents/sany/work/big_data_geology/geochemistrypy/criteria/docstring/test.py", line 19, in __main__.foo
Failed example:
    foo(2, 7)
Expected:
    7
Got:
    9
Trying:
    foo(1, 2)
Expecting:
    3
ok
1 items had no tests:
    __main__
**********************************************************************
1 items had failures:
   1 of   2 in __main__.foo
2 tests in 2 items.
1 passed and 1 failed.
***Test Failed*** 1 failures.
```

### (2) Unittest

+ Test samples are stored in a separate file outside the module to be tested, unlike Doctest, which is stored in the module's Docstring.
+ It is not as easy to use as the Doctest module, but it can provide a more comprehensive set of tests in a separate file.
+ samples:

tbc...



## 7. Git Commit Message

+ One commit for one functionality implementation or one optimization or one bug fix. It is allowed to push multiple commits to the remote codebase with one pull request.
+ Please include the following tags in the beginning of your commit message to make more organized commits and PRs. It would tell exactly what use it is in this commit.
  + `feat`: a new feature is introduced with the changes
  + `fix`: a bug fix
  + `perf`: an existing feature improved
  + `docs`: changes to the documentation
  + `style`: code formatting
  + `refactor`: refactoring production code
  + `revert`:  version revertion
  + `chore`: the change of developing tools or assisted tool
  + `test`: adding missing tests, refactoring tests
  + `build`: package the codebase
  + `ci`: continue integration
  + `BREAKING CHANGE`: a breaking API change

Notice: the tags should be lower-case except for `BREAKING CHANGE`.

+ Sample:

```
# Assume that Sany build up a function of drawing the decision boundary for Decision Tree.
# Then he wants to use git commit to append the code changes to his local codebase.
# He will use the command in command line.

git commit -m "feat: add a function to draw the decision boundary for Decision Tree."
```

+ Reference:
  + [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.2/#specification)
