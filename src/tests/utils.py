from utils import log_or_print


import sys
import traceback
from utils import (
    log_or_print
)
from logger import make_logger

def test_wrapper(func):

    def wrapped_func(*args, **kwargs):
        logger = make_logger()
        try:
            test_name = str(func).split()[1]
            log_or_print(
                "Starting test \"{}\"".format(test_name),
                logger,
                msg_type="info"
            )
            func(*args, **kwargs)
            log_or_print(
                f"{test_name}: OK",
                logger,
                msg_type="log"
            )
        except:
            log_or_print(
                f"{test_name}: Fail",
                logger,
                msg_type="error"
            )
            log_or_print(
                f"Reason: \n{traceback.format_exc()}",
                logger,
                msg_type="error"
            )
            sys.exit(1)

    return wrapped_func
