class Error(Exception):
    pass

class NoDatasetError(Error):
    "Dataset can't be found."
    pass

class InvalidCommandError(Error):
    "Provided command is invalid."
    pass 

class QCFractalError(Error):
    "QCFractal error"
    pass

class QCEngineError(Error):
    "QCEngine error"
    pass

class OptimizeInputError(Error):
    "Optimize input file error"
    pass

class OptimizationError(Error):
    "Optimization failed"
    pass

class WorkflowError(Error):
    "TRIC workflow failed or stopped"
    pass
