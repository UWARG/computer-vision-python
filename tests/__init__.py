def _import_all_modules():
    """ Dynamically imports all modules in this package. """
    import traceback
    import os
    global __all__
    __all__ = []
    globals_, locals_ = globals(), locals()

    # get DeepSort submodules
    os.chdir('modules/targetAcquisition/Yolov5_DeepSort_Pytorch')
    os.system('git submodule update --init')
    os.chdir('../../../')


    # Dynamically import all the package modules in this file's directory.
    for filename in os.listdir(f'{__name__}'):
        # Process all python files in directory that don't start
        # with underscore (which also prevents this module from
        # importing itself).
        if filename[0] != '_' and filename.split('.')[-1] in ('py', 'pyw'):
            module_name = filename.split('.')[0]  # Filename sans extension.
            package_module = '.'.join([__name__, module_name])
            try:
                module = __import__(package_module, globals_, locals_, [module_name])
            except:
                traceback.print_exc()
                raise
            for name in module.__dict__:
                if not name.startswith('_'):
                    globals_[name] = module.__dict__[name]
                    __all__.append(name)


_import_all_modules()