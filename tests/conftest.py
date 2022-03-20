def pytest_addoption(parser):
    parser.addoption('--integration', action='store_true', dest="integration",
                 default=False, help="enable longrundecorated tests")