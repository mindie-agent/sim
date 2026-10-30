#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""
op_compiler_parse_args
"""
import ast
import os
import getopt
import json
import sys
import multiprocessing
from typing import List, Tuple, Dict

from tbe.common.utils import log as logger
from tbe.common.utils.AscendLog import AscendLog
from op_compiler_common import OPTOptions
from op_compiler_common import SOC_TO_SHORT_SOC_MAP
from op_compiler_common import COMPILING_OPTION_KEYS
from op_compiler_common import Parameter_Mapping
from op_compiler_common import time_stamp
from op_compiler_common import verify_pathfile_exist_permission
from op_compiler_common import judgment_option_exist
from op_compiler_common import LOG_LEVEL_DICT

# 运行时禁止生成__pycache__/文件夹
sys.dont_write_bytecode = True

EXIT_CODE_0 = 0
EXIT_CODE_1 = 1
EXIT_CODE_NONE = None


class Option:
    """
    define a option.
    no_args means the option does not need a value. e.g.
    -h or --help
    """

    def __init__(self, name, default_value="", help_message="", no_args=False):
        """
        init class
        """
        self.__name = name
        self.__default_value = default_value
        self.__help_message = help_message
        self.__no_args = no_args
        self.__value = default_value

    def get_name(self) -> str:
        """
        return the name of this option
        """
        return self.__name

    def get_value(self) -> str:
        """
        return the value of this option
        """
        return self.__value

    def set_value(self, value):
        """
        Set the value of this option
        """
        self.__value = value


class OptionParser:
    """
    This class with parse the user's input arguments
    with getopt.
    """

    def __init__(self):
        """
        init class
        """
        self.full_options = {}
        self.compiling_options = {}
        self.__short_options = ""
        self.__long_options = []

    @staticmethod
    def usage():
        """
        print the help message
        """
        print("usage: op_compiler <args>\n"
              "example of compiling :\t"
              "op_compiler --op_params_dir=./input --output=./output_dir --soc_version=Ascend910B1\n"
              "============================== Basic Functionality ==============================\n"
              "[General]\n"
              "  -h/--help                show help message.\n"
              "  -l/--log                 setting log level, ['DEBUG', 'INFO', 'WARNING', 'ERROR'].\n"
              "  -p/--op_params_dir       path of the parameter file exported by the Dump tool.\n"
              "  --count                  count the number of JSON files, used with [-p/--op_params_dir].\n"
              "  --enable_super_kernel    enable super kernel.\n"
              "  --super_kernel_options   super kernel options.\n"
              "  --super_kernel_optimize_options   super kernel optimize options as a JSON object "
              "or Python dict literal.\n"
              "  --super_kernel_debug_options      super kernel debug options as a JSON object "
              "or Python dict literal.\n"
              "  -o/--output              output directory of installation package.\n"
              "  -k/--kernel_name         operator kernel name.\n"
              "  -d/--op_debug_config     configuring CCEC compilation options.\n"
              "  -v/--soc_version         soc version of current hardwar.\n"
              "  -m/--compile_mode        operator compile mode, default is opc mode. [tune]/[aoe] is being sunset "
              "and will be automatically switched to opc mode.\n"
              "  -f/--follow_op_params_jit    whether follow all op params setting or not,"
              " default is [false], optional [true].\n"
              "  -j/--job                 number of processes.")

    def __define_option(self, real_name_list: list, default_value: str, help_message: str, no_args: bool=False):
        """
        :params: real_name_list:
                 default_value:
                 help_message:
                 no_args:
        """
        for real_name in real_name_list:
            option_object = Option(real_name, default_value, help_message, no_args)
            # full name contains dash, for example
            # option "help" has full name "--help"
            # option "h" has full name "-h"
            if len(real_name) == 1:
                self.__short_options = self.__short_options + real_name
                if not no_args:
                    self.__short_options = self.__short_options + ":"
                full_name = "-" + real_name
            else:
                name_with_equal = real_name
                if not no_args:
                    name_with_equal = name_with_equal + "="
                self.__long_options.append(name_with_equal)
                full_name = "--" + real_name
            self.full_options[full_name] = option_object

    def define_options(self):
        """
        Register all options with:
        (option key, default_value, help message, do not need value).
        """
        self.__define_option([OPTOptions.HELP, OPTOptions.HELP_H], \
            None, "show help messages", True)
        self.__define_option([OPTOptions.OP_PARAMS_DIR, OPTOptions.OP_PARAMS_DIR_P], \
            None, "path of the parameter file exported by the Dump tool")
        self.__define_option([OPTOptions.COUNT], \
            None, "count json number", True)
        self.__define_option([OPTOptions.ENABLE_SUPER_KERNEL], \
            None, "enable super kernel", True)
        self.__define_option([OPTOptions.SUPER_KERNEL_OPTIONS], \
            None, "super kernel options")
        self.__define_option([OPTOptions.SUPER_KERNEL_OPTIMIZE_OPTIONS], \
            None, "super kernel optimize options as a JSON object or Python dict literal")
        self.__define_option([OPTOptions.SUPER_KERNEL_DEBUG_OPTIONS], \
            None, "super kernel debug options as a JSON object or Python dict literal")
        self.__define_option([OPTOptions.SOC_VERSION, OPTOptions.SOC_VERSION_V], \
            None, "soc version of current hardwar")
        self.__define_option([OPTOptions.OUTPUT, OPTOptions.OUTPUT_O], \
            None, "output directory of binary files")
        self.__define_option([OPTOptions.KERNEL_NAME, OPTOptions.KERNEL_NAME_K], \
            None, "operator kernel name")
        self.__define_option([OPTOptions.OP_TYPE, OPTOptions.OP_TYPE_T], \
            None, "operator type")
        self.__define_option([OPTOptions.OP_DEBUG_CONFIG, OPTOptions.OP_DEBUG_CONFIG_D], \
            None, "configuring CCEC Compilation Optionss")
        self.__define_option([OPTOptions.COMPILE_MODE, OPTOptions.COMPILE_MODE_M], \
            None, "operator compile mode")
        self.__define_option([OPTOptions.FOLLOW_OP_PARAMS, OPTOptions.FOLLOW_OP_PARAMS_F], \
            f"false", "whether follow all op params setting or not")
        self.__define_option([OPTOptions.JOB, OPTOptions.JOB_J], \
            f"16", "number of processes")
        self.__define_option([OPTOptions.LOG, OPTOptions.LOG_L], \
            "ERROR", "log level")

        self.full_options["--output_path"] = Option(OPTOptions.OUTPUT_PATH, "", "", )
        self.full_options["--package_name"] = Option(OPTOptions.PACKAGE_NAME, "", "", )

    @staticmethod
    def strip_key(key: str) -> str:
        """
        original key is --xxx or -x, we get rid of -- or - in this function
        :params: input key of option
        :return: ret_key
        """
        length = len(key)
        if length == 2:
            ret_key = key[1:]
        elif length >= 3:
            ret_key = key[2:]
        else:
            ret_key = key
        return ret_key

    def __set_full_options_value(self, options: list, value: str):
        """
        :params: options, value
        """
        for option in options:
            self.full_options[option].set_value(value)

    def __get_real_package_name(self, package_name: str=None) -> str:
        """
        :params: package_name: 传入的run包名
        :return: 需要生成的run包名
        """
        if package_name:
            if package_name.endswith('.run'):
                return package_name
            else:
                return package_name + '.run'

        formatted_time = time_stamp()
        if "d" in self.compiling_options or "op_debug_config" in self.compiling_options:
            return 'debug_kernel_{}.run'.format(formatted_time[:12])
        else:
            return 'static_kernel_{}_{}.run'.format(formatted_time, os.getpid())

    def __verify_output(self, path: str=None) -> Tuple[bool, str, str]:
        """
        check whether the output is valid
        :params: path
        :return: True/False, out_path, package_name
        """
        out_path, package_name = None, None
        if path:
            parent_path = os.path.dirname(path)
            if path.endswith(os.sep): # 以os.sep结尾的为路径，否则可能为package_name
                if not verify_pathfile_exist_permission(path, "path", "e").exist_state:
                    logger.error(f"[{path}] path does not exist. Please check.")
                    return False, out_path, package_name
                if not verify_pathfile_exist_permission(path, "path", "w").w_permission_state:
                    logger.error(f"[{path}] do not have the write permission., please setting.")
                    return False, out_path, package_name
                out_path = path
                package_name = self.__get_real_package_name()
            elif os.sep not in path: # 输入path直接为package name
                out_path = os.getcwd()
                package_name = self.__get_real_package_name(path)
            else:
                if not os.path.exists(path):
                    if not os.path.exists(parent_path): # 检查父路径是否存在
                        logger.error(f"[{parent_path}] does not exist. please check.")
                        return False, out_path, package_name
                    else:
                        if not os.access(parent_path, os.W_OK):
                            logger.error(f'[{parent_path}] do not have the write permission., please setting.')
                            return False, out_path, package_name
                        else: # 父路径存在且有权限
                            out_path = parent_path
                            package_name = self.__get_real_package_name(os.path.basename(path))
                else:
                    out_path = path
                    package_name = self.__get_real_package_name()
        else: # --output参数不存在
            out_path = os.getcwd()
            package_name = self.__get_real_package_name()
        logger.debug(f'out path is : {out_path}')
        logger.debug(f'package_name is : {package_name}')
        return True, out_path, package_name

    @staticmethod
    def __verify_soc_version(soc_version: str) -> bool:
        """
        check whether the SOC version are supported.
        :params: soc_version
        :return: True/False
        """
        low_soc_version = soc_version.lower()
        if low_soc_version not in SOC_TO_SHORT_SOC_MAP:
            logger.error(f"input soc verison [{soc_version}] is not supported. please check.")
            return False
        return True

    @staticmethod
    def __verify_compile_mode(compile_mode: str) -> bool:
        """
        Check whether the compile mode is supported. [tune]/[aoe] are accepted
        for backward compatibility but are being sunset and will be redirected
        to opc mode at compile time.
        :params: compile_mode
        :return: True/False
        """
        if compile_mode not in ["", "opc", "tune", "aoe"]:
            logger.error(f"compile mode supports only [opc] or [tune] or [aoe] or blank. please check.")
            return False
        return True

    @staticmethod
    def __verify_follow_op_params(follow_op_params_switch: str) -> bool:
        """
        Check whether the compile mode is tune.
        :params: compile_mode
        :return: True/False
        """
        if follow_op_params_switch not in ["false", "true"]:
            logger.error(f"follow op params must be false or true. please check.")
            return False
        return True

    def __verify_log_level(self, log_level: str) -> bool:
        """
        Check whether the log level.
        :return: True/False
        """
        if log_level.lower() not in ["debug", "info", "warning", "error"]:
            print(f"[ERROR]: input log level error: {log_level}.")
            return False
        return True

    @staticmethod
    def set_log_level(log_level_str):
        """
        Set the environment argument of log.
        :param log_level_str:
        :return:
        """
        log_level = LOG_LEVEL_DICT.get(log_level_str, 3)
        ascend_logger = AscendLog()
        # 57 is TBE, 8 is CCE
        ascend_logger.set_level(57, log_level, 0)
        ascend_logger.set_level(8, log_level, 0)

    def update_log_options(self, opts: list):
        """
        update log options
        """
        for key, value in opts:
            if key in ("-l", "--log"):
                if not self.__verify_log_level(value):
                    return False
                self.__set_full_options_value(["-l", "--log"], value)
                self.set_log_level(value.lower())
        return True

    def get_cmd_options(self, opts: list):
        """
        parsing all parameters in the command line
        """
        for key, value in opts:
            striped_key = self.strip_key(key)
            if striped_key in COMPILING_OPTION_KEYS:
                try:
                    self.compiling_options[striped_key] = value
                except KeyError as e:
                    logger.error(e)

    def parse_output_info(self, opts: list):
        """
        parsing output_path and package_name
        """
        if "o" in self.compiling_options or "output" in self.compiling_options:
            for key, value in opts:
                if key in ("-o", "--output"):
                    flag, output_path, package_name = self.__verify_output(value)
                    if not flag:
                        return False
                    self.__set_full_options_value(["--output_path"], output_path)
                    self.__set_full_options_value(["--package_name"], package_name)
                    self.__set_full_options_value(["-o", "--output"], value)
        else:
            flag, output_path, package_name = self.__verify_output()
            self.__set_full_options_value(["--output_path"], output_path)
            self.__set_full_options_value(["--package_name"], package_name)
        return True

    def check_op_params_dir(self, op_params_dir: str) -> bool:
        """
        check op_params_dir is valid.
        """
        for _, _, files in os.walk(op_params_dir):
            for file in files:
                if file.endswith('.json'):
                    return True
        return False

    @classmethod
    def __format_super_kernel_option_value(cls, value) -> str:
        """
        Convert a JSON option value to the legacy super kernel option format.
        """
        if isinstance(value, dict):
            sub_items = [
                f"{key}={cls.__format_super_kernel_option_value(sub_value)}"
                for key, sub_value in value.items()
            ]
            return ",".join(sub_items)
        if isinstance(value, list):
            return ",".join(str(item) for item in value)
        return str(value)

    @staticmethod
    def __parse_super_kernel_structured_option(option_name: str, option_value: str):
        """
        Parse one SuperKernel JSON/Python option and require a dictionary.
        """
        try:
            options = json.loads(option_value)
        except (json.JSONDecodeError, TypeError):
            try:
                options = ast.literal_eval(option_value)
            except (ValueError, SyntaxError, TypeError) as err:
                logger.error(
                    f"[{option_name}] must be a valid JSON object or Python "
                    f"dict literal: {err}"
                )
                return False, None
        if not isinstance(options, dict):
            logger.error(
                f"[{option_name}] must be a JSON object or Python dict literal, but got "
                f"{type(options).__name__}."
            )
            return False, None
        return True, options

    def __normalize_super_kernel_structured_options(self, opts: list) -> bool:
        """
        Normalize JSON/Python options to --super_kernel_options.
        """
        option_keys = {key for key, _ in opts}
        structured_option_names = (
            "--super_kernel_optimize_options",
            "--super_kernel_debug_options",
        )
        present_structured_options = [
            option_name
            for option_name in structured_option_names
            if option_name in option_keys
        ]
        if not present_structured_options:
            return True
        if "--super_kernel_options" in option_keys:
            logger.error(
                "[--super_kernel_options] and "
                "[--super_kernel_optimize_options/--super_kernel_debug_options] "
                "cannot be used together."
            )
            return False
        if "--enable_super_kernel" not in option_keys:
            logger.error(
                "[--super_kernel_optimize_options/--super_kernel_debug_options] "
                "must be used with [--enable_super_kernel]."
            )
            return False

        normalized_parts = []
        normalized_keys = set()
        for option_name in present_structured_options:
            option_value = self.full_options[option_name].get_value()
            status, option_dict = self.__parse_super_kernel_structured_option(
                option_name, option_value
            )
            if not status:
                return False
            duplicate_keys = normalized_keys.intersection(option_dict)
            if duplicate_keys:
                duplicate_key = sorted(duplicate_keys)[0]
                logger.error(
                    f"[{option_name}] contains duplicate option [{duplicate_key}] "
                    "across SuperKernel option groups."
                )
                return False
            normalized_keys.update(option_dict)
            normalized_parts.extend(
                f"{key}={self.__format_super_kernel_option_value(value)}"
                for key, value in option_dict.items()
            )

        self.__set_full_options_value(
            ["--super_kernel_options"], ":".join(normalized_parts)
        )
        return True

    def parse_other_options(self, opts: list):
        """
        parsing other options info
        """
        for key, value in opts:
            if key in ("-p", "--op_params_dir"):  # 校验op_params_dir输入路径是否存在且有效
                if not verify_pathfile_exist_permission(value, "path", "e").exist_state or \
                    not self.check_op_params_dir(value):
                    logger.error(f"[{value}] path does not exist or is not valid. Please check.")
                    return False
                self.__set_full_options_value(["-p", "--op_params_dir"], value)
            if key in ("-t", "--op_type"):
                self.__set_full_options_value(["-t", "--op_type"], value)
            if key in ("-k", "--kernel_name"):
                self.__set_full_options_value(["-k", "--kernel_name"], value)
            if key in ("-d", "--op_debug_config"):
                if not verify_pathfile_exist_permission(value, "file", "e").exist_state:
                    logger.error(f"[{value}] file does not exist. Please check.")
                    return False
                if not verify_pathfile_exist_permission(value, "file", "r").r_permission_state:
                    logger.error(f"[{value}] do not have the read permission., please setting.")
                    return False
                self.__set_full_options_value(["-d", "--op_debug_config"], value)
            if key in ("-v", "--soc_version"):
                if not self.__verify_soc_version(value):
                    return False
                self.__set_full_options_value(["-v", "--soc_version"], value)
            if key in ("-m", "--compile_mode"):
                if not self. __verify_compile_mode(value):
                    return False
                self.__set_full_options_value(["-m", "--compile_mode"], value)
            if key in ("-f", "--follow_op_params_jit"):
                if not self. __verify_follow_op_params(value):
                    return False
                self.__set_full_options_value(["-f", "--follow_op_params_jit"], value)
            if key in ("-j", "--job"):
                self.__set_full_options_value(["-j", "--job"], value)
            if key in ("--count"):
                self.__set_full_options_value(["--count"], None)
            if key in ("--enable_super_kernel"):
                self.__set_full_options_value(["--enable_super_kernel"], True)
            if key in ("--super_kernel_options"):
                self.__set_full_options_value(["--super_kernel_options"], value)
            if key in ("--super_kernel_optimize_options"):
                self.__set_full_options_value(
                    ["--super_kernel_optimize_options"], value
                )
            if key in ("--super_kernel_debug_options"):
                self.__set_full_options_value(
                    ["--super_kernel_debug_options"], value
                )
        return self.__normalize_super_kernel_structured_options(opts)

    def _parse_verify_options(self, opts: list) -> int:
        """
        parse options
        :return: exit_code
        """
        # 更新full_options
        if not self.update_log_options(opts):
            return EXIT_CODE_1

        # 获取命令行中所有的编译选项
        self.get_cmd_options(opts)

        # 检查是否有help选项
        if "h" in self.compiling_options or "help" in self.compiling_options:
            self.usage()
            return EXIT_CODE_0

        # 校验必填参数
        if "count" not in self.compiling_options  and "v" not in self.compiling_options \
            and "soc_version" not in self.compiling_options:
            logger.error("[-v/--soc_version] is mandatory, please check.")
            return EXIT_CODE_1

        # 根据output 解析出 output_path, package_name
        if not self.parse_output_info(opts):
            return EXIT_CODE_1

        # 遍历校验其余入参
        if not self.parse_other_options(opts):
            return EXIT_CODE_1
        return EXIT_CODE_NONE

    def get_options(self) -> int:
        """
        parse options using getopt.
        :return: exit_code
        """
        # 无参数和参数为-h/--help
        if len(sys.argv) <= 1:
            print("[ERROR]: args is empty. Please enter correct compiling arguments.")
            self.usage()
            return EXIT_CODE_1

        if len(sys.argv) == 2:
            arg = sys.argv[1]
            if arg in ("-h", "--help"):
                self.usage()
                return EXIT_CODE_0

        # 获取并解析所有命令行参数
        try:
            opts, args = getopt.getopt(sys.argv[1:], self.__short_options, self.__long_options)
        except getopt.GetoptError as err:
            print("[ERROR]: failed to get Args.")
            self.usage()
            return EXIT_CODE_1
        exit_code = self._parse_verify_options(opts)

        if exit_code == EXIT_CODE_1 or exit_code == EXIT_CODE_0:
            return exit_code
        logger.info(f"compiling args is {self.compiling_options}")
        for option in self.full_options.values():
            logger.debug(f"full option : [{option.get_name()}], value : [{option.get_value()}]")

        return EXIT_CODE_NONE
