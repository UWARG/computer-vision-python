# Coding Style Guide

This is the style guide for CV, following [PEP 8](https://peps.python.org/pep-0008/) and [Pylint](https://pylint.org/). Follow this guide for all new code (you do not need to worry about old code). Use common sense and/or mention @Xierumeng on Discord if there is something missing.

## Pylint

Pylint can either be installed as a [Visual Studio Code extension](https://marketplace.visualstudio.com/items?itemName=ms-python.pylint) or through `pip` . Fix lint issues before asking for review, as otherwise the PR might get delayed as the reviewer will ask you to resolve those issues before it can be merged.

If a lint issue cannot be resolved by restructuring code, Pylint can be disabled for a specific line with:

```python
# pylint: disable=[problem-code]
```

All Pylint disables MUST be accompanied by a comment explaining why this is necessary.

## Basics

There is an example in [General](#general) below.

* `variable_names` in snake_case
* `function_names()` in snake_case
* `ClassNames` in PascalCase (UpperCamelCase)
* `CONSTANT_NAMES` in CAPITAL_SNAKE_CASE
* `file_names` in snake_case
* Private members (functions and variables) should have two underscores to indicate them as such: `def __my_private_func():` , `__my_private_set = set()`
* Initialize any variables within class constructor to `None` or some other value
* Use constants for numbers that can be tweaked (e.g. pixel height/pixel width of an image, number of epochs in a model)
* 4 spaces per level of indentation
* There should be no space before an opening parantheses (`my_function(myVar)` NOT `my_function (myVar)` )
* Only use parantheses when necessary (`while myInt < 3:` NOT `while (myInt < 3):` )
* Operators:
    * No spaces around `*` , `/` , `%` , `!` , `:` if it's used as a slicing operator
    * One space on either side of `=`, `==`, `+`, `-`, `+=`, `-=`, etc
        * Except in named arguments of a function call (e.g. `np.sum(arr, axis=1)` )
    * One space after every comma: `my_func(var1, var2, var3)`
* 160 character limit per line (not a hard limit, use common sense)

## Files

Module/file names are in snake_case: `geolocation_worker.py`

File contents end with 1 blank line:

```python
# Stuff
# Last blank line below, then end of file

```

Indents are 4 spaces long. Blank lines do not have any spaces. Strings are encoded with UTF-8 and enclosed with **double quotes**.

## Imports

Import system-level modules, then installed modules, then local (CV) modules. Separate each group with 1 blank line, with module names in alphabetical order within the group:

```python
import os
import sys

from math import pi
import numpy as np

import modules.geolocation
```

Do not use wildcards (asterisk/star) in imports, except for the following modules:

```python
from typing import *
```

Do not call private functions from an imported module:

```python
import some_cv_module

some_cv_module.__some_private_function()  # NO
```

## General

It's easier to show by example for reference than giving a bunch of forgettable rules:

```python
import os
import sys

from math import pi
import numpy as np
from typing import *


# Top-level constants are in UPPER_SNAKE_CASE
# Do not use top-level mutables (all top-level variables are constants ONLY)
UPPER_SNAKE_CASE = True
# 1 space after the octothorpe/number sign/hash in a comment
GLOBAL_CONST = 5  # 2 spaces before an inline comment
# 2 blank lines between top-level imports, constants, function definitions, and class definitions


# Types from other modules include the prefix (e.g. `np.ndarray` ), except from the typing module
# Containers (e.g. list, dict, tuple, set) include the type of the enclosed objects
def get_coordinates(new_frame: np.ndarray=np.zeros((416, 416, 3))) -> Dict[yolov2_assets.utils.BoundBox, Tuple[int, int]]:
    """
    Function, method, and variable names are in snake_case.
    Function and method parameters have type hints where possible.

    Every function, class, and method has a docstring below the signature (NOT above):

    * Provide a short description of what the function/method/class does.
    * End each sentence in a docstring with punctuation (period, colon, etc.).
    * Docstring text on a new line (triple quotes are by themselves on a separate line).

    Optionally, add additional documentation:

    1. Description of what the function does.

    2. A list of parameters, including name, type, whether it is optional, and a description of the parameter.

    3. The function's return type (if any), as well as a description of what it returns.

    In general, the code itself should be easy to follow so that not much documentation is required.
    """
    # Example docstring:
    """
    Returns a list of co-ordinates along a video frame where tents are located by running YOLOV2 model.

    Parameters
    ----------
    new_frame: np.ndarray, optional
        Variable size array containing data about a video frame (as given by cv2.imread()).

    Returns
    -------
    Dict[yolov2_assets.utils.BoundBox, Tuple[int, int]]
        Returns a dict of bounding boxes where tents are located, with centre coordinates for each.
    """
    pass


def __some_private_function() -> float:
    """
    Top-level private functions have 2 underscores as prefix.
    """
    return 0.5


# Class names are CamelCase
class VideoDecoder:
    """
    Decodes video into individual frames.

    Attributes
    ----------
    encoding: str
    quality: float
    __effort: int
    __resolution: float

    Methods
    -------
    __init__()
        Sets decoding settings.
    decode_frame()
        Decodes a frame from a video file.
    __h264_decode()
        Decode an H.264 video.
    __static_helper()
        There is nothing stopping you from copy-pasting the method description from its docstring.
    """
    # 1 blank line between class methods

    def __init__(self, encoding: str, quality: float):
        """
        `__init__` is the class constructor.

        All members are first created here. None are created outside this method!
        Public members and methods follow snake_case, as normal.
        Private members and methods have 2 underscores as prefix.
        Use common sense when determining which members should be public vs private.
        """
        self.encoding = encoding
        self.quality = quality
        self.__effort = None  # Use None if you want to create this member but save initialization for later
        self.__resolution = quality * 400


    def decode_frame(self, video: str) -> Tuple[bool, List[List[int]]]:
        """
        The first parameter of class methods is `self` , all other parameters are after, except for static methods (see below).
        """
        # Prefer using explicit type construction over implicit (e.g. dict() over {} ), with the exception of lists (square brackets).
        some_dictionary = dict()
        self.__h264_decode(some_dictionary)
        return True, [[0]]

    def __h264_decode(self, arg0: Dict[str, int]):
        """
        Method that returns nothing.
        """
        return

    @staticmethod
    def __static_helper(dividend: int, divide_by_ten: bool) -> int:
        """
        Static class methods are pure functions: They have an input, do something, and have an output.
        Pure functions do not save state, which is why the `self` parameter is omitted.
        In technical terms, this is described as having no side effects.

        Do not use static class members (for the same reason global variables are not allowed).
        Static members appear as variables within the class but outside of any method (static or otherwise).
        There are none in the VideoDecoder example class, so it's safe to use as a reference.
        """
        quotient = dividend
        # Conditionals and loops are not surrounded by parantheses
        if divide_by_ten:
            quotient = dividend / 10
            # 1 blank line after the end of the block

        return quotient
        # 1 blank line at end of file

```

## Multiline

Function and method definitions (calls have a different style below):

```python
def example_function(arg0: int, arg1: int,
                     arg2: int, arg3: int,
                     arg4: int):
    """
    Example.
    """
    pass
```

Conditionals and loops use parantheses:

```python
if (condition0 and
    condition1):
    # Required comment for separation of condition and body
    pass

while (conditio0 and
       condition1):
    # Required comment for separation of loop and body
    pass
```

Backslash: At the end of the line, 1 space followed by backslash:

```python
# Any required string encoding uses UTF-8
with open("/path/to/some/file/you/want/to/read", encoding="utf-8") as file_1, \
     open("/path/to/some/file/being/written", "w", encoding="utf-8") as file_2:
    pass
```

Everything else:

```python
example0 = example_function(
    0, 1,
    2, 3,
    4
)

example1 = dict(
    name0 = [
        [0, 0],
        [1, 1],
        [2, 2]
    ],
    name1 = [
        [3, 3]
    ]
)

example2 = [
    [
        [0, 0],
        set("hi", "hie", "fie"),
        dict(
            a = 1,
            b = 2
        )
    ]
]
```

# Suggestions

## Exceptions

Avoid throwing and catching exceptions, as it is very slow. Instead, use return checking:

```python
def possibly_failing_function(num: int) -> Tuple[bool, int]:
    """
    Fails if num is zero.
    """
    if num == 0:
        return False, None

    # Do some stuff
    return True, 3 / num


values = []
for i in range(0, 10):
    result, value = possibly_failing_function(3)
    if not result:
        continue

    values.append(value)
```

## Pythonic

Follow a Pythonic style of coding if possible. For example:

Iterating:

```python
file_names = [
    "a",
    "b",
    "c",
    "d"
]

for file_name in file_names:
    print(file_name)  # This will print `a`, then `b`, then `c`, etc.
```

Slicing and matrix multiplication:

```python
# 3x2 array
a = np.array(
    [
        [1, 1],
        [2, 3],
        [5, 8]
    ]
)

a_sum_columns = a.sum(axis=0)
# array([8, 12])

a_sum_rows = a.sum(axis=1)
# array([2, 5, 13])

a_sum_all0 = a.sum()
# 20

a_sum_all1 = a.sum(axis=None)
# 20

# @ is the matrix multiplication operator
a @ a_sum_columns
# array([20, 52, 136])

# Get column 1
a[:, 1]
# array([1, 3, 8])
```

Zipping:

```python
my_keys = ["C", "V", "P"]
my_values = [3, 5, 0]

dict(zip(my_keys, my_values))
# Dict[str, int]: {"C": 3, "V": 5, "P": 0}
```
