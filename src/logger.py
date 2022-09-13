from utils import (
    get_current_time
)
import os


PROGRESS_FREQUENCY = 0.01
INDENT = "    "


# string style
CONTROL_PREFIX = "\033["
CONTROL_SUFFIX = "m"
CONTROL_SEPARATOR = ";"
DEFAULT_TEXT_STYLE = "0"
BOLD_TEXT_STYLE = "1"
BLACK_COLOR_CODE = "40"
RED_COLOR_CODE = "31"
GREEN_COLOR_CODE = "32"
BLUE_COLOR_CODE = "34"
PURPLE_COLOR_CODE = "35"
CYAN_COLOR_CODE = "36"
WHITE_COLOR_CODE = "37"
GREY_COLOR_CODE = "0"


# logger messages
LOG_PREFIX = "(log)"
INFO_PREFIX = "(info)"
ERROR_PREFIX = "(error)"
MAX_LINE_LENGTH = 80
SEPARATOR = "".join(["="] * MAX_LINE_LENGTH)


def make_string_style(
    text_style,
    text_color,
    background_color=BLACK_COLOR_CODE
):
    return (
        f"{CONTROL_PREFIX}{text_style}{CONTROL_SEPARATOR}"
            + f"{text_color}{CONTROL_SEPARATOR}"
            + f"{background_color}{CONTROL_SUFFIX}"
        )


class RedneckLogger:
    def __init__(self, output_folder=None):
        if output_folder:
            self.update_output_folder(output_folder)
        else:
            self.stdout_file = None
            self.stderr_file = None
        self.cache = {}

    def store(self, name, msg):
        assert isinstance(msg, str)
        self.cache[name] = msg

    def dump(self, name):
        self.log('\n' + self.cache[name])

    def get_output_folder(self):
        return os.path.dirname(self.stdout_file)

    def update_output_folder(self, new_output_folder):
        os.makedirs(new_output_folder, exist_ok=True)
        self.stdout_file = os.path.join(new_output_folder, "stdout.txt")
        self.stderr_file = os.path.join(new_output_folder, "stderr.txt")

    def log_separator(self):

        print(SEPARATOR)

        if self.stdout_file:
            print(
                SEPARATOR,
                file=open(self.stdout_file, "a"),
                flush=True
            )

    def progress(
        self,
        descripion,
        current_step,
        total_steps,
        frequency=PROGRESS_FREQUENCY
    ):
        log_every_n_steps = \
            max(1, round(PROGRESS_FREQUENCY * total_steps)) \
                if frequency is not None \
                else 1
        if (
            current_step % log_every_n_steps == 0
                or current_step == 1
                or current_step == 2
                or current_step == total_steps
        ):
            self.log(
                "{} {}/{}: {}/100%..".format(
                    descripion,
                    current_step,
                    total_steps,
                    round(100 * float(current_step / total_steps))
                ),
                carriage_return=(current_step != total_steps)
            )

    def log(
        self,
        msg,
        auto_newline=False,
        carriage_return=False
    ):

        self.logger_output(
            msg,
            LOG_PREFIX,
            prefix_style_code=make_string_style(
                BOLD_TEXT_STYLE,
                GREEN_COLOR_CODE
            ),
            message_style_code=make_string_style(
                BOLD_TEXT_STYLE,
                WHITE_COLOR_CODE
            ),
            output_file=self.stdout_file,
            auto_newline=auto_newline,
            carriage_return=carriage_return
        )

    def info(
        self,
        msg,
        auto_newline=False,
        carriage_return=False
    ):

        info_style_code = make_string_style(
            BOLD_TEXT_STYLE,
            PURPLE_COLOR_CODE
        )

        self.logger_output(
            msg,
            INFO_PREFIX,
            prefix_style_code=info_style_code,
            message_style_code=info_style_code,
            output_file=self.stdout_file,
            auto_newline=auto_newline,
            carriage_return=carriage_return
        )

    def error(
        self,
        msg,
        auto_newline=False,
        carriage_return=False
    ):

        self.logger_output(
            msg,
            ERROR_PREFIX,
            prefix_style_code=make_string_style(
                BOLD_TEXT_STYLE,
                RED_COLOR_CODE
            ),
            message_style_code=make_string_style(
                BOLD_TEXT_STYLE,
                WHITE_COLOR_CODE
            ),
            output_file=self.stderr_file,
            auto_newline=auto_newline,
            carriage_return=carriage_return
        )

    def logger_output(
        self,
        msg,
        prefix_keyword,
        prefix_style_code,
        message_style_code,
        output_file=None,
        auto_newline=False,
        carriage_return=False
    ):
        msg_prefix = "{} {}".format(
            get_current_time(),
            prefix_keyword
        )
        end_char = '' if carriage_return else '\n'
        print(
            self.make_log_message(
                msg,
                msg_prefix,
                prefix_style_code=prefix_style_code,
                message_style_code=message_style_code,
                auto_newline=auto_newline,
                carriage_return=carriage_return
            ),
            flush=True,
            end=end_char
        )

        if output_file:
            print(
                self.make_log_message(
                    msg,
                    msg_prefix,
                    prefix_style_code="",
                    message_style_code="",
                    auto_newline=auto_newline,
                    carriage_return=carriage_return
                ),
                file=open(output_file, "a"),
                flush=True,
                end=end_char
            )

    def make_log_message(
        self,
        msg,
        prefix,
        prefix_style_code="",
        message_style_code="",
        auto_newline=False,
        carriage_return=False
    ):
        outside_style_code = ""
        if prefix_style_code:
            assert message_style_code
            outside_style_code = make_string_style(
                DEFAULT_TEXT_STYLE,
                WHITE_COLOR_CODE,
                GREY_COLOR_CODE
            )
        return insert_char_before_max_width(
            "{}{}: {}{}{}".format(
                prefix_style_code,
                prefix,
                message_style_code,
                msg,
                outside_style_code
            ),
            MAX_LINE_LENGTH if auto_newline else 0
        ) + ('\r' if carriage_return else '')


def make_logger(output_folder=None):
    return RedneckLogger(output_folder)


def insert_char_before_max_width(
    input_string,
    max_width,
    char='\n',
    separator=" ",
    indent=INDENT
):
    if len(input_string) == 0 or max_width == 0:
        return input_string
    current_line = ""
    result = ""
    for word in input_string.split(separator):
        if current_line == "":
            current_line = word
        elif (len(current_line) + len(word) <= max_width):
            current_line = current_line + separator + word
        else:
            result += current_line + char
            current_line = indent + word
    result += current_line
    return result
