# Bug Template
(1) System Environment:
+ Systerm version: macOS Catalina version 10.15.7
+ Python version: 3.9.7
+ Package version(Optional): geopandas-0.10.2

(2)	Problem:
when I run >>> python3 main.py, I got a problem with the package geopandas-0.10.2. I succesfully download it through pip3. However, when I import it, it fails.

(3)	Log:
Successfully installed
```
➜  client git:(main) ✗ pip3 install geopandas
Collecting geopandas
  Using cached geopandas-0.10.2-py2.py3-none-any.whl (1.0 MB)
Requirement already satisfied: pyproj>=2.2.0 in /usr/local/lib/python3.9/site-packages (from geopandas) (3.0.1)
Requirement already satisfied: shapely>=1.6 in /usr/local/lib/python3.9/site-packages (from geopandas) (1.7.1)
Requirement already satisfied: pandas>=0.25.0 in /Users/can/Library/Python/3.9/lib/python/site-packages (from geopandas) (1.4.1)
Requirement already satisfied: fiona>=1.8 in /usr/local/lib/python3.9/site-packages (from geopandas) (1.8.19)
Requirement already satisfied: certifi in /usr/local/lib/python3.9/site-packages (from pyproj>=2.2.0->geopandas) (2020.11.8)
Requirement already satisfied: numpy>=1.18.5; platform_machine != "aarch64" and platform_machine != "arm64" and python_version < "3.10" in /usr/local/lib/python3.9/site-packages (from pandas>=0.25.0->geopandas) (1.19.4)
Requirement already satisfied: python-dateutil>=2.8.1 in /usr/local/lib/python3.9/site-packages (from pandas>=0.25.0->geopandas) (2.8.1)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.9/site-packages (from pandas>=0.25.0->geopandas) (2020.4)
Requirement already satisfied: cligj>=0.5 in /usr/local/lib/python3.9/site-packages (from fiona>=1.8->geopandas) (0.7.1)
Requirement already satisfied: click<8,>=4.0 in /usr/local/lib/python3.9/site-packages (from fiona>=1.8->geopandas) (7.1.2)
Requirement already satisfied: click-plugins>=1.0 in /usr/local/lib/python3.9/site-packages (from fiona>=1.8->geopandas) (1.1.1)
Requirement already satisfied: attrs>=17 in /usr/local/lib/python3.9/site-packages (from fiona>=1.8->geopandas) (20.3.0)
Requirement already satisfied: six>=1.7 in /Users/can/Library/Python/3.9/lib/python/site-packages (from fiona>=1.8->geopandas) (1.16.0)
Requirement already satisfied: munch in /usr/local/lib/python3.9/site-packages (from fiona>=1.8->geopandas) (2.5.0)
Installing collected packages: geopandas
Successfully installed geopandas-0.10.2
```
However import fails.
```
➜  client git:(main) ✗ pip3 show geopandas
WARNING: Package(s) not found: geopandas
```
In Pythoin Environment:
```python
Python 3.9.7 (default, Oct 13 2021, 06:44:56)
[Clang 12.0.0 (clang-1200.0.32.29)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import geopandas
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'geopandas'

```
(4)	Idea:
Probably, there is some problem about the way to install geopandas or Python Version.

(5)	Contact:
Email: sanyhew1097618435@163.com
