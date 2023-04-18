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

1. In the computer vision repository, run the command: `virtualenv [name]` , where `[name]` is the virtual environment name. You can also type `virtualenv -p python3 [name]` if you really want to make sure it's Python3.

    - Use `venv` or `env` for the name (do not use anything else). These instructions will use `venv`

1. Now a virtualenv called `venv` is created. Activate the virtual environment.

    * Windows cmd: `venv\Scripts\Activate.bat`
    * Windows PowerShell: `.\venv\Scripts\Activate.ps1`
        * If your system says `Execution of script is not available` , open PowerShell as Administrator and run `powershell Set-ExecutionPolicy RemoteSigned`
    * Mac or Linux: `source venv/bin/activate`

1. You will then see `(venv)` in front of your command line, which means you are in the virtual environment.

1. Install the required packages:

    1. `pip install -r requirements.txt`
    1. `pip install -r requirements-pytorch.txt`
    1. Do NOT use `requirements-jetson.txt` , that is for the Jetson only.
    1. Do NOT use `requirements-old.txt` , that is from 2022 (old).

1. To exit the virtual environment: `deactivate`

## Git Practices

To implement a new feature:

1. Create a new branch from main: `git checkout -b my-new-branch-name`

1. Add functional changes in one or more commits (ideally, each commit should be targeted to developing a specific feature/editing a certain file).

1. Develop and commit test cases before submitting a request for review. We use Pytest for unit testing.

1. Add documentation in a separate commit, up to standards outlined below.

1. Add formatting fixes in a separate commit.

1. Any bug fixes in a separate commit, with a well-named message.

1. **Pull from main to get branch up to date, then commit this change**.

1. Submit a pull request on the GitHub webpage with detailed message outlining:

    * Description of purpose of PR
    * New changes
    * Outstanding changes
    * Any other comments

## Coding Style

**Readable code is important**! Follow the [CV Coding Style Guide](STYLE.md) .
