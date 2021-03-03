# Computer Vision Python

A new computer vision system written in Python

Please read the documentation and git best practices below before creating a pull request

## Git Practices
To implement a new feature:
1. Create a new branch from main
2. Add functional changes in one or more commits (ideally, each commit should be targeted to developing a specific feature/editing a certain file)
3. Develop and commit test cases before submitting a request for review
4. Add documentation in a separate commit, up to standards outlined below
5. Add formatting fixes in a separate commit
6. Any bug fixes in a separate commit, with a well-named message
7. **Pull from main to get branch up to date, commit this change**
8. Submit pull request with detailed message outlining:
  * Description of purpose of PR
  * New changes
  * Outstanding changes
  * Any other comments

## Naming and typing conventions
* `variableNames` in camelCase
* `function_names()` in snake_case
* `ClassNames` in PascalCase (UpperCamelCase)
* `CONSTANT_NAMES` in CAPITAL_SNAKE_CASE
* `FileNames` in PascalCase (UpperCamelCase)
* "Private" members (functions & variables) should have two underscores to indicate them as such: `def __myPrivateFunc()`, `__myPrivateSet = set()`
* Initialize any variables within class constructor to `None`, or some other value
* Use constants for numbers that can be tweaked (e.g. pixel height/pixel width of an image, number of epochs in a model)
* 4 spaces per level of indentation
* There should be no space before an opening bracket (`myFunction(myVar)` NOT `myFunction (myVar)`)
* Only use brackets when necessary (`while myInt < 3` NOT `while (myInt < 3)`)
* Operators:
* No spaces around `*`, `/`, `%`, `!`
* One space on either side of `=`, `==`, `+`, `-`, `+=`, `-=`, etc
* One space after every comma `myFunc(var1, var2, var3)`
* Import statments should be grouped in the following order:
1. Standard library imports (os, json, sys)
2. Third-party imports (tensorflow, scipy, numpy - anything you need pip to install)
3. Local file imports (modules.TargetAcquisition, YOLOv2Assets)
* Try to keep imports in alpha order
```
import os
import sys

import tensorflow as tf
import numpy as np

import modules.TargetAcquisition
```
* 160 character limit per line (not a hard limit, use common sense)
* Hanging indents should be aligned to delimeter:
```
my_function(hasToo,
            many, variables)
```

## Docstrings
General guidelines:
* Single line comments explain your code, and should be created with `#` followed by a space:
```
# This comment explains what the following code does
```
* Classes and functions need to have appropriate multiline docstrings, created with triple double-quotes:
```
"""
This comment explains a function/class
They can be several lines long
If the comment is only one line, it is acceptable to put the comment and the triple double-quotes on one line
"""
```

### Files
* Writing file headers is not necessary

### Functions
Function annotations should include:
1. Description of what the function does. If the function is a private member function, write the word `PRIVATE: ` in capital letters before describing the function's purpose.
2. A list of parameters, including name, type, whether it is optional, and a description of the parameter.
3. The function's return type (if any), as well as a description of what it returns
The function annotation should come after the declaration, indented as if it were part of the code.
Example:
```
def get_coordinates(self, newFrame=np.zeros((416, 416, 3))):
    """
    Returns a list of co-ordinates along a video frame where tents are located by running YOLOV2 model
    
    Parameters
    ----------
    newFrame : np.ndarray, optional
        Variable size array containing data about a video frame (as given by cv2.imread())

    Returns
    -------
    dict<yolov2_assets.utils.BoundBox : tuple<xCo : int, yCo : int>>
        Returns a dict of bounding boxes where tents are located, with centre coordinates for each
    """
    # Function definition here
```
Notice that the type descriptions are very specific. When declaring a type, use the following conventions:
* Inbuilt types, self defined types, and classes can just have the class name listed. Be descriptive as to where the type is defined (e.g. `np.ndarray`, NOT `ndarray`)
* Complex types that hold other data (e.g. `dict`s, `set`s, `list`s, etc) should indicate what type of data is being stored within them by using angle brackets `<>` (e.g. `dict<myClass : list<int>`
For information on a list of Python inbuilt datatypes, see https://www.w3schools.com/python/python_datatypes.asp

### Classes
Class annotations should include:
1. Description of what the class does
2. Attributes, including names, types, and descriptions
3. Methods, including names, parameters (names and types), and descriptions
Class annotations should come after the class definition, indented as if it were part of the code
Example:
```
class Line:
    """
    Holds information about a line in 2D space and functions that can be performed on it.
    
    Attributes
    ----------
    __point1 : tuple<xCo: int, yCo: int>
    __point2 : tuple<xCo: int, yCo: int>
    
    Methods
    -------
    __init__(p1 : tuple<xCo: int, yCo: int>, p2: tuple<xCo: int, yCo: int>)
        Creates a line given two points
    print()
        Prints coordinates of the points that define the line
    __draw()
        Draws the line on a canvas
    set_p1(newPoint: tuple<xCo: int, yCo: int>)
        Sets the point coordinates of __point1
    get_p1()
        Returns location of __point1
    set_p2(newPoint: tuple<xCo: int, yCo: int>)
        Sets the point coordinates of __point2
    get_p2()
        Returns location of __point2
    """
    # Class code below
```
Note that although the above class makes all of its attributes private and uses getters and setters, this is not always needed for our purposes. Use common sense when determining what members should be public vs private.

Also note that it isn't necessary to explain which functions/attributes are public vs private here, the naming convention to use two underscores (`__myPrivateVar`) is enough to convey this information.

### More docs
If you encounter a scenario that isn't mentioned above, refer to the Python PEP8 style guide (https://www.python.org/dev/peps/pep-0008/#function-annotations) or use common sense.
