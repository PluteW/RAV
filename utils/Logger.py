import logging
import os
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

import torch.distributed as dist
from tensorboardX import SummaryWriter

from Config.Config import ConfigObject

logger_initialized = {}


def getLoggerAndWritter(bsconfig: ConfigObject):

    # MTAG For log files
    assert hasattr(bsconfig, "logpath"), "Miss the logpath in config.yaml file"
    logpath = bsconfig.logpath
    if not os.path.exists(logpath):
        os.makedirs(logpath)

    # logdir_name = time.strftime("%Y%m%d-%H%M", time.localtime())
    # logfileDirPath = logpath + "/" + bsconfig.logname
    # if not os.path.exists(logfileDirPath):
    #     os.makedirs(logfileDirPath)

    # dir_num = len(os.listdir(logfileDirPath))
    # logfilepathdir = logfileDirPath + f"/run{dir_num+1}"

    # dir_num = len(os.listdir(logpath))
    # logfilepathdir = logpath + f"/run{dir_num+1}"
    logfilepathdir = logpath

    if not os.path.exists(logfilepathdir):
        os.makedirs(logfilepathdir)
    logfilepath = logfilepathdir + "/runtime.log"

    # # MTAG config backup
    # backConfigPath = logfilepathdir + f"/{bsconfig.logname}.yaml"
    # saveConfigToYaml(config, backConfigPath)

    log_level = logging.INFO
    if hasattr(bsconfig, "loglevel"):
        if bsconfig.loglevel == "DEBUG" or bsconfig.loglevel == "debug":
            log_level = logging.DEBUG
        elif bsconfig.loglevel == "INFO" or bsconfig.loglevel == "info":
            log_level = logging.INFO
        elif bsconfig.loglevel == "WARN" or bsconfig.loglevel == "warn":
            log_level = logging.WARN
        elif bsconfig.loglevel == "ERROR" or bsconfig.loglevel == "error":
            log_level = logging.ERROR

    assert hasattr(bsconfig, "logname")

    logger = getRootLogger(
        log_file=logfilepath, name=bsconfig.logname, log_level=log_level
    )

    # train_writer = SummaryWriter(os.path.join(logfilepathdir, "train"))
    # val_writer = SummaryWriter(os.path.join(logfilepathdir, "test"))
    # printLog(f"Starting Training: {config.statement}", logger)
    # printLog(f"Config has been save in {backConfigPath} .", logger)
    printLog(f"Log will be save in {logfilepathdir}", logger)

    # return logger, train_writer, val_writer
    return logger

def getLoggerObject(logpath:str, loglevel:str, logname:str):

    
    if not os.path.exists(logpath):
        os.makedirs(logpath)

    # logdir_name = time.strftime("%Y%m%d-%H%M", time.localtime())
    # logfileDirPath = logpath + "/" + bsconfig.logname
    # if not os.path.exists(logfileDirPath):
    #     os.makedirs(logfileDirPath)

    # dir_num = len(os.listdir(logfileDirPath))
    # logfilepathdir = logfileDirPath + f"/run{dir_num+1}"

    # dir_num = len(os.listdir(logpath))
    # logfilepathdir = logpath + f"/run{dir_num+1}"
    logfilepathdir = logpath

    if not os.path.exists(logfilepathdir):
        os.makedirs(logfilepathdir)
    logfilepath = logfilepathdir + "/runtime.log"

    # # MTAG config backup
    # backConfigPath = logfilepathdir + f"/{bsconfig.logname}.yaml"
    # saveConfigToYaml(config, backConfigPath)

    log_level = logging.INFO
   
    if loglevel == "DEBUG" or loglevel == "debug":
        log_level = logging.DEBUG
    elif loglevel == "INFO" or loglevel == "info":
        log_level = logging.INFO
    elif loglevel == "WARN" or loglevel == "warn":
        log_level = logging.WARN
    elif loglevel == "ERROR" or loglevel == "error":
        log_level = logging.ERROR

    logger = getRootLogger(
        log_file=logfilepath, name=logname, log_level=log_level
    )

    # train_writer = SummaryWriter(os.path.join(logfilepathdir, "train"))
    # val_writer = SummaryWriter(os.path.join(logfilepathdir, "test"))
    # printLog(f"Starting Training: {config.statement}", logger)
    # printLog(f"Config has been save in {backConfigPath} .", logger)
    printLog(f"Log will be save in {logfilepathdir}", logger)

    # return logger, train_writer, val_writer
    return logger


def getRootLogger(log_file=None, log_level=logging.INFO, name="main"):
    """Get root logger and add a keyword filter to it.
    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
        name (str, optional): The name of the root logger, also used as a
            filter keyword.
    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = getLogger(name=name, log_file=log_file, log_level=log_level)
    # add a logging filter
    logging_filter = logging.Filter(name)
    logging_filter.filter = lambda record: record.find(name) != -1

    return logger


def getLogger(name, log_file=None, log_level=logging.INFO, file_mode="w"):
    """Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.
    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.
    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # handle duplicate logs to the console
    # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
    # to the root logger. As logger.propagate is True by default, this root
    # level handler causes logging messages from rank>0 processes to
    # unexpectedly show up on the console, creating much unwanted clutter.
    # To fix this issue, we set the root logger's StreamHandler, if any, to log
    # at the ERROR level.
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger


def printLog(msg, logger=None, level=logging.INFO):
    """Print a log message.
    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == "silent":
        pass
    elif isinstance(logger, str):
        _logger = getLogger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            "logger should be either a logging.Logger object, str, "
            f'"silent" or None, but got {type(logger)}'
        )


logger = getLoggerObject("/home/aa/Desktop/WJL/VTRAG/logging", "INFO", "runtime.log")