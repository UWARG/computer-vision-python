# Computer Vision Python

A new computer vision system written in Python.

Please read the documentation and Git best practices below before creating a pull request.

## Setup

1. Configure Git with your credentials, name, and email.

    - **Ensure `core.autocrlf` is set to `true` or `input` ! Line endings on GitHub should be LF only, not CRLF**!

1. Find a directory on your PC and `git clone` this repository.

    - `~/documents/: git clone [link]` will create a directory called `computer-vision-python/` : `~/documents/computer-vision-python/`

1. `cd` into the newly created directory.

1. Clone Git submodules: `git submodule update --init --recursive` and `git submodule update --remote`

1. Download and install Python 3.8 (any sub-version should do but we're using 3.8.10 if you want to be exact).

1. Create a virtual environment by first installing `virtualenv` : `pip install virtualenv`

1. In the computer vision repository, run the command: `virtualenv cv_python` , where `cv_python` is the virtual environment name. You can also type `virtualenv -p python3 cv_python` if you really want to make sure it's Python3.

1. Now a virtualenv called `cv_python` is created.

    * Windows: `.\cv_python\Scripts\Activate.ps1` (in case your system says `Execution of script is not available` , open Powershell as Administrator and run `powershell Set-ExecutionPolicy RemoteSigned` ).

    * Mac or Linux: `source ./cv_python/bin/activate`

1. You will then see a `(cv_python)` in front of your command line, which means you are in the virtual environment.

1. `pip install -r requirements.txt` will automaticaly install everything you need.

## Git Practices

To implement a new feature:

1. Create a new branch from main: `git checkout -b my-new-branch-name`

2. Add functional changes in one or more commits (ideally, each commit should be targeted to developing a specific feature/editing a certain file).

3. Develop and commit test cases before submitting a request for review. We use Pytest for unit testing.

4. Add documentation in a separate commit, up to standards outlined below.

5. Add formatting fixes in a separate commit.

6. Any bug fixes in a separate commit, with a well-named message.

7. **Pull from main to get branch up to date, then commit this change**.

8. Submit a pull request on the GitHub webpage with detailed message outlining:

    * Description of purpose of PR
    * New changes
    * Outstanding changes
    * Any other comments

## Coding Style

**Readable code is important**! Follow the [CV Coding Style Guide](STYLE.md) .
