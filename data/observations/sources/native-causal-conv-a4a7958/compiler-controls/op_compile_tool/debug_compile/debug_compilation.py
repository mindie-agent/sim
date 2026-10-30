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

import os
import sys
import json
import shutil
import itertools  
import subprocess
from copy import deepcopy
from typing import List, Tuple, Dict
from collections import namedtuple

from tbe.common.utils import log as logger
from opc_tool.opc_api import compile_op
from op_compiler_parse_args import OptionParser
from debug_compile.get_dyn_support_info import init_dyn_binary_info
from debug_compile.get_dyn_support_info import get_dyn_binpath_by_kernelname
from debug_compile.get_dyn_support_info import  get_binary_info_by_binpath                                              
from debug_compile.gen_debug_binary_info_config import gen_debug_binary_info_config
from static_compile.gen_binary_info_config import gen_binary_info_config
from static_compile.get_support_info import get_static_binary_info_by_kernelname
from op_compiler_common import time_stamp
from op_compiler_common import trans_soc_ver_to_short
from op_compiler_common import verify_pathfile_exist_permission
from op_compiler_common import make_dirs
from op_compiler_common import remove_tempfile
from op_compiler_common import generate_version_info
from op_compiler_common import get_opp_path
from op_compiler_common import path_conversion
from op_compiler_common import corrections_op_type
from op_compiler_common import get_makeself_file


# 运行时禁止生成__pycache__/文件夹
sys.dont_write_bytecode = True

DEBUG_COMPILATION_PATH = os.path.abspath(os.path.dirname(__file__))


def _parse_op_debug_config(path: str) -> Tuple[bool, str]:
    """
    parse the debug config content
    :param path: path
    :return: True/False, op_debug_config
    """
    if not verify_pathfile_exist_permission(path, "file", "e").exist_state:
        logger.error(f"[{path}] path does not exist. Please check.")
        return False, None
    if not verify_pathfile_exist_permission(path, "file", "r").r_permission_state:
        logger.error(f"[{path}] do not have the read permission., please setting.")
        return False, None
    try:
        with open(path, mode='r', encoding='utf-8') as fd:
            op_debug_config = fd.readline().split("=")[-1].strip()
            logger.debug(f"the op_debug_config file is opened successfully, {path}, [{op_debug_config}]")
    except Exception as e:
        logger.exception(f"{path}, failed to open the op_debug_config file")
        return False, None
    op_debug_config_list  = op_debug_config.split(",")
    for item in op_debug_config_list: # 检查op_debug_config是否合法
        if item not in ["oom", "dump_bin", "dump_cce", "dump_loc", "ccec_O0", "ccec_g", "check_flag"]:
            logger.error(f"op_debug_config value {item} is invalid.")
            return False, None
    return True, op_debug_config   


def _parse_bin_path_for_josn(dir_: str, file_: str) -> Tuple[bool, str]:
    """
    :params: dir
             file_
    :return: True/False, bin_path
    """

    op_binary_json_path = os.path.join(dir_, file_)
    if not verify_pathfile_exist_permission(op_binary_json_path, "file", "e").exist_state and \
        not verify_pathfile_exist_permission(op_binary_json_path, "file", "r").r_permission_state:
        logger.error(f"[{op_binary_json_path}] does not exist or do not have the read permission.")
        return False, None, None
    try:
        with open(op_binary_json_path, mode='r', encoding='utf-8') as fd:
            op_binary = json.load(fd)
    except Exception as e:
        logger.exception(f"{op_binary_json_path}, failed to open the op_debug_config file.")
        return False, None, None
    if "bin_info" not in op_binary:
        return False, None, None
    if op_binary['bin_info'].endswith(".o"):
        bin_path = op_binary['bin_info']
    else:
        bin_path = op_binary['bin_info'].replace(".json", ".o")
    json_path = bin_path.replace(".o", ".json")
    if not verify_pathfile_exist_permission(bin_path, "file", "e").exist_state \
        and not verify_pathfile_exist_permission(bin_path, "file", "r").r_permission_state:
        logger.error(f"[{op_binary['bin_info']}] does not exist or do not have the read permission.")
        return False, None, None
    if not verify_pathfile_exist_permission(json_path, "file", "e").exist_state \
        and not verify_pathfile_exist_permission(json_path, "file", "r").r_permission_state:
        logger.error(f"[{op_binary['bin_info']}] does not exist or do not have the read permission.")
        return False, None, None
    return True, bin_path, json_path


def _get_json_path_list(op_params_dir: str) -> list:
    """
    Oobtains all bin paths
    :param path: op_params_dir
    :return: bin_path_list
    """
    json_path_list = []
    for root, _, files in os.walk(op_params_dir):
        for file_ in files:
            if not file_.endswith('.json'):  
                continue
            flag, bin_path, json_path = _parse_bin_path_for_josn(root, file_)
            if not flag:
                return False, None
            json_path_list.append(json_path)
            logger.debug(f"bin_path {bin_path} is obtained successfully from {os.path.join(root, file_)}")
    return True, json_path_list


def _opc_compile_single(op_params, build_options):
    """
    compiling the OPC of a single operator
    :param path: op_params, build_options
    """
    logger.info(f"operator [{op_params['op_type']}] opc compilation started.")
    logger.debug(f"op_params : {op_params}")
    logger.debug(f"build_options : {build_options}")
    if compile_op(op_params=op_params, build_options=build_options):
        logger.info(f"operator [{op_params['op_type']}] Opc compile success.")
    else:
        logger.error(f"operator [{op_params['op_type']}] Opc compile failed.")


def _wrapper_func(args: tuple):
    """
    function wrapper
    :param func, op_params, build_options
    """
    func, a, b = args
    return func(a, b)


def _create_operator_package_dir(dir_list: list) -> bool:
    """
    :param dir_list
    :return: True/False
    """
    for dir_ in dir_list:
        if not make_dirs(dir_):
            return False
    logger.debug("operator package path is successfully constructed")
    return True


def _copy_bin_json_file(op_name: str, op_binary_info: dict, binarylist_key: str, \
                        compile_res_dir: str, target_path: str):
    """
    :params: op_name
             op_binary_info
             binarylist_key
             compile_res_dir
             target_path
    :return: True/False
    """    
    for binary_info in op_binary_info[binarylist_key]:
        compile_res_bin_path = os.path.join(compile_res_dir, os.path.basename(binary_info["binPath"]))
        compile_res_json_path  = os.path.join(compile_res_dir, os.path.basename(binary_info["jsonPath"])) 
        try:
            shutil.copy(compile_res_bin_path, os.path.join(target_path, op_name))
            shutil.copy(compile_res_json_path, os.path.join(target_path, op_name))
        except Exception as e:
            logger.exception(f"failed to copy binary files {compile_res_bin_path}.")
            return False 
    return True  


def _copy_kernel_binary_files(binary_info_config_json: dict, compile_res_dir :str, target_path: str) -> bool:
    """
    :param binary_info_config_json
    :param compile_res_dir
    :param target_path
    :return: True/False
    """
    for op_type, op_binary_info in binary_info_config_json.items():
        if "binaryList" in op_binary_info:
            op_name = os.path.basename(os.path.dirname(op_binary_info["binaryList"][0]["binPath"]))     
        else:
            op_name = os.path.basename(os.path.dirname(op_binary_info["staticList"][0]["binPath"]))  
        make_dirs(os.path.join(target_path, op_name))
        for binarylist_key in ["binaryList", "staticList"]:
            if binarylist_key not in op_binary_info:
                continue
            if not _copy_bin_json_file(op_name, op_binary_info, binarylist_key, compile_res_dir, target_path):
                return False
        logger.debug(f"{op_type} binary files is successfully copied to [{os.path.join(target_path, op_name)}]")
    logger.info(f"binary files is successfully copied to {target_path}")
    return True


def _write_new_op_binary_json(binary_params) -> bool:
    """
    :params: op_type
             op_name
             op_binary_info
             soc_version
             dyn_binary_path
             new_op_binary_json_path
    :return: True/False
    """
    new_op_binary_json = {"binList": []}
    new_op_binary_json_path = os.path.join(binary_params.config_socversion_path, f"{binary_params.op_name}.json")
    for binary_info in binary_params.op_binary_info["binaryList"]:
        json_path = binary_info["jsonPath"]
        op_binary_json_path = os.path.join(binary_params.dyn_binary_path, 
                                           "config", binary_params.soc_version, f"{binary_params.op_name}.json")
        try:
            with open(op_binary_json_path, mode='r', encoding='utf-8') as fd:
                op_binary_json = json.load(fd)
        except Exception as e:
            logger.exception(f"failed to open {op_binary_json_path}")
            return False
        for op_binary in op_binary_json["binList"]:
            if op_binary["binInfo"]["jsonFilePath"] == json_path:
                new_op_binary_json["binList"].append(op_binary)
    try:
        with open(new_op_binary_json_path, 'w') as file_path:
            json.dump(new_op_binary_json, file_path, sort_keys=True, indent=2, separators=(',', ':'))
    except Exception as e:
            logger.exception(f"failed to write {new_op_binary_json_path}")
            return False
    logger.debug(f"{binary_params.op_type} binary files is successfully copied to [{new_op_binary_json_path}]")
    return True


def _copy_op_binary_files(dyn_binary_path :str, binary_info_config_json: dict, \
                          soc_version: str, config_socversion_path: str) -> bool:
    """
    :params: binary_info_config_json
    :return: True/False
    """
    for op_type, op_binary_info in binary_info_config_json.items():
        if "binaryList" in op_binary_info:
            op_name = os.path.basename(os.path.dirname(op_binary_info["binaryList"][0]["binPath"])) 
        else:
            op_name = os.path.basename(os.path.dirname(op_binary_info["staticList"][0]["binPath"])) 
        if "binaryList" in op_binary_info:
            binary_params = namedtuple("binary_params", \
                                       "op_type, op_name, op_binary_info, soc_version, \
                                        dyn_binary_path, config_socversion_path")
            if not _write_new_op_binary_json(binary_params(op_type, \
                                                           op_name, op_binary_info, soc_version, \
                                                           dyn_binary_path, config_socversion_path)):
                return False
    logger.info(f"op binary info files is successfully copied to {config_socversion_path}")  
    return True


def _copy_scripts(debug_kernel_path: str, scripts_path: str) -> bool:
    """
    :param debug_kernel_path
    :param scripts_path
    :return: True/False
    """
    try:
        for src_file in ["common.py", "install.py", "uninstall.py", "help.info"]:
            shutil.copy(os.path.join(DEBUG_COMPILATION_PATH, "scripts", src_file), scripts_path)
            os.chmod(os.path.join(scripts_path, src_file), 0o755)
        shutil.copy(os.path.join(DEBUG_COMPILATION_PATH, "scripts", "uninstall.sh"), debug_kernel_path)
        os.chmod(os.path.join(debug_kernel_path, "uninstall.sh"), 0o755)
    except Exception as e:
        logger.exception(f"failed to copy scripts files.")
        return False
    logger.info(f"scripts files is successfully copied to [{debug_kernel_path}]")
    return True


def _create_operator_package(package_params) -> bool:
    """
    :params: soc_version: 芯片名代号
             package_name: 包名
             output_path: 产物输出路径
             compile_res_dir: opc编译产物生成路径
             binary_json_file: binary_info_config.json 路径
             binary_info_config_json: 
             dyn_binary_path: 
    :return: True/False
    """
    delivery_path = os.path.join(package_params.output_path, "delivery")
    operator_package_path = os.path.join(delivery_path, 'operator_package')
    debug_kernel_path = os.path.join(operator_package_path, 'debug_kernel')
    scripts_path = os.path.join(debug_kernel_path, "scripts")
    debugkernel_soc_version_path = os.path.join(debug_kernel_path, package_params.soc_version)
    config_path = os.path.join(debug_kernel_path, "config")
    config_socversion_path = os.path.join(config_path, package_params.soc_version)

    # 创建run包路径结构
    if not _create_operator_package_dir([delivery_path, operator_package_path, debug_kernel_path, \
        scripts_path, debugkernel_soc_version_path, config_path, config_socversion_path]):
        return False

    # 复制.o和.json到debug_kernel/$package_params.soc_version 中
    if not _copy_kernel_binary_files(package_params.binary_info_config_json, \
                                     package_params.compile_res_dir, debugkernel_soc_version_path):
        return False

    # 复制binary_info_config.json 到目标路径
    try:
        shutil.copy(package_params.binary_json_file, config_socversion_path)
        logger.debug(f"binary_info_config.json file is successfully copied to [{config_socversion_path}]")
    except Exception as e:
        logger.exception(f"failed to copy binary_info_config.json.")
        return False
    # 复制dyn下config/$package_params.soc_version 到对应的debug路径下
    if not _copy_op_binary_files(package_params.dyn_binary_path, \
                                 package_params.binary_info_config_json, \
                                 package_params.soc_version, config_socversion_path):
        return False

    # 拷贝安装卸载脚本
    if not _copy_scripts(debug_kernel_path, scripts_path):
        return False

    # 生成version.info
    flag, version, version_dir = generate_version_info()
    if flag:
        with open(f'{debug_kernel_path}/version.info', mode='wt', encoding='utf-8') as file:
            file.write(version)
            file.write(version_dir)
        logger.debug("version information of the opp package is successfully written to debug package.")
    else:
        logger.error("failed to write the version information of the opp package to debug package.")
        return False
    
    return True


def _set_dir_mod(ascend_path: str, scene: str, original_permissions=288) -> tuple:
    """
    :param ascend_path: 变量的路径
    :param scene: before_makeself/after_makeself
    :param original_permissions: 目录原权限
    :return: (bool, original_permissions)
    """
    if scene == 'before_makeself':
        original_permissions = (os.stat(ascend_path).st_mode & 0o777)
        os.chmod(ascend_path, 0o755)
    elif scene == 'after_makeself':
        os.chmod(ascend_path, original_permissions)
    else:
        logger.error('scene error')
        return False, ''
    return True, original_permissions


def compose_makeself_command(ascend_opp_path: str, package_name: str, output_path: str) -> tuple:
    """
    组装makeself包打包命令
    :param ascend_opp_path: 环境变量
    :param package_name: run包名
    :param output_path: run包输出路径
    :return: makeself包打包命令, ascend_opp_path, output_path目录权限
    """
    _, opp_original_permissions = _set_dir_mod(ascend_opp_path, 'before_makeself')

    makeself_tool, makeself_header = get_makeself_file(ascend_opp_path)
    if not makeself_tool or not makeself_header:
        return None, f"makeself_tool or makeself_header does not exist.", ''

    help_info = os.path.join(output_path, 'delivery', 'operator_package', 'debug_kernel', 'scripts', 'help.info')
    install_script = os.path.join('debug_kernel', 'scripts', 'install.py')

    comments = '"DEBUG KERNEL RUN PACKAGE"'
    commands = itertools.chain(
        [makeself_tool, "--header", makeself_header, "--help-header", help_info, '--complevel', '4',
         '--nomd5', '--sha256', '--nooverwrite', '--chown', '--tar-format', 'gnu',
         '--tar-extra', '--numeric-owner', '--tar-quietly'
         ],
        [f'{output_path}/delivery/operator_package', package_name, comments, install_script]
    )

    command = ' '.join(commands)
    return command, '', opp_original_permissions


def _create_run_package_command(ascend_opp_path: str, package_name: str, output_path: str) -> Tuple[bool, str]:
    """
    :params: package_name: run包名
             output_path: run包输出路径
    :return: True/False, data
    """
    makeself_cmd, data, opp_original_permissions = compose_makeself_command(ascend_opp_path,
                                                                            package_name,
                                                                            output_path)
    if not makeself_cmd:
        logger.error(data)
        return False, ''
    run_package_cmd = f'cd {output_path} && {makeself_cmd}'
    logger.debug(f'run_package_cmd: {run_package_cmd}')
    status, data = subprocess.getstatusoutput(run_package_cmd)
    _set_dir_mod(ascend_opp_path, 'after_makeself', opp_original_permissions)
    if status != 0:
        return False, data
    return True, data
    

def _print_debug_compiler_params(debug_compiler_params):
    """
    prints all input parameters.
    """   
    logger.debug(f"op_params_dir : {debug_compiler_params.op_params_dir}")
    logger.debug(f"kernel_name : {debug_compiler_params.kernel_name}")
    logger.debug(f"op_debug_config_path : {debug_compiler_params.op_debug_config_path}")
    logger.debug(f"package_name : {debug_compiler_params.package_name}")
    logger.debug(f"output_path : {debug_compiler_params.output_path}")
    logger.debug(f"log_level : {debug_compiler_params.log_level}")
    logger.debug(f"short_soc_version : {debug_compiler_params.short_soc_version}")


def _modify_support_info_format(params: dict):
    """
    :params: params: dict
    """
    for item in params:
        if item and "dtype" in item and item["dtype"] == "float":
            item["dtype"] = "float32"


def _modify_support_info(support_info: dict) -> dict:
    """
    :params: support_info
    :return: support_info
    """
    for param in ['inputs', 'outputs', 'attrs']:
        if param not in support_info:
            continue
        if isinstance(support_info[param], list):
            _modify_support_info_format(support_info[param])
        else:
            _modify_support_info_format(support_info[param])
    return support_info


def _assemble_opc_params(soc_version, compile_res_dir, op_debug_config, log_level, json_path_list: list) -> list:
    """
    :params: soc_version: 
             compile_res_dir: opc编译结果输出路径
             op_debug_config:
             log_level:
             json_path_list:
    :return: opc编译命令
    """
    op_param_list = []

    op_params_template = {}
    build_options_template = {
        "soc_version" : soc_version,
        "output" : compile_res_dir,
        "op_debug_config": op_debug_config,
        "log": log_level
    }
    for json_path in json_path_list:
        op_params = deepcopy(op_params_template)
        build_options = deepcopy(build_options_template)

        # 使用json_path获取binary_json_info
        flag, binary_json_info = get_binary_info_by_binpath(json_path)
        if not flag:
            logger.error(f"failed to search for binary_json_info for json_path {json_path}.")
            continue
        try:
            # 获取support_info
            support_info = binary_json_info["supportInfo"]
            if "implMode" in support_info:
                support_info["impl_mode"] = support_info["implMode"]
            support_info = _modify_support_info(support_info)

            support_info["bin_filename"] = binary_json_info["binFileName"]
            op_type = corrections_op_type(binary_json_info["binFileName"].split("_")[0])
            # 拼装编译参数
            op_params["op_type"] = op_type
            op_params["op_list"] = [support_info]
            build_options["bin_filename"] = binary_json_info["binFileName"]
            if "opMode" in support_info:
                build_options["op_mode"] = support_info["opMode"]
            op_param_list.append((_opc_compile_single, op_params, build_options))
        except Exception as e:
            logger.exception(f"Parsing and compiling from {json_path} failed.")
    return op_param_list


def _check_opc_result(json_path_list, compile_res_dir) -> Tuple[bool, list]:
    """
    :params: compile_res_dir: opc编译结果输出路径
    :return: True/False, real_json_path_list
    """
    real_json_path_list = []
    for json_path in json_path_list:
        bin_name = os.path.splitext(os.path.basename(json_path))[0] + ".o"
        if verify_pathfile_exist_permission(os.path.join(compile_res_dir, bin_name), "file", "e").exist_state:
            real_json_path_list.append(json_path)
        else:
            logger.warn(f"{bin_name} compilation result does not exist.")
    if len(real_json_path_list) != len(json_path_list):
        return False, None
    return True, json_path_list


def _init_related_path() -> Tuple[bool, str, str]:
    """
    :return: True/False, ascend_opp_path, dyn_binary_path
    """
    ascend_opp_path = None
    dyn_binary_path = None

    relative_dyn_binary_path = "built-in/op_impl/ai_core/tbe/kernel"
    flag, ascend_opp_path = get_opp_path()
    logger.info(f"ascend opp path is {ascend_opp_path}")
    if not flag:
        return False, ascend_opp_path, dyn_binary_path
    dyn_binary_path = os.path.join(ascend_opp_path, relative_dyn_binary_path)
    if not verify_pathfile_exist_permission(dyn_binary_path, "path", "e").exist_state \
        and not verify_pathfile_exist_permission(dyn_binary_path, "path", "r").r_permission_state:
        logger.error(f"[{dyn_binary_path}] does not exist or do not have the read permission")
        return False, ascend_opp_path, dyn_binary_path
    logger.info(f"run dynamic binary path is {dyn_binary_path}")
    return True, ascend_opp_path, dyn_binary_path


def assemble_compile_param(assemble_compile_params: list):
    op_params_dir, kernel_name, short_soc_version, dyn_binary_info, soc_version, \
    compile_res_dir, op_debug_config, log_level = assemble_compile_params
    json_path_list = []
    if op_params_dir:
        get_json_path_list_flag, json_path_list = _get_json_path_list(os.path.abspath(op_params_dir))
        if not get_json_path_list_flag or not json_path_list:
            logger.error(f"failed to parse bin_path from the {op_params_dir} file.")
            return None, None           
    elif kernel_name:
        binary_info_flag, json_path, binary_json_info = \
            get_static_binary_info_by_kernelname(kernel_name, short_soc_version)
        if not binary_info_flag:
            binary_info_flag, bin_path, json_path = \
                get_dyn_binpath_by_kernelname(kernel_name, dyn_binary_info, short_soc_version)
        if not binary_info_flag:
            logger.error(f"failed to search for bin path for kernelname {kernel_name}.")
            return None, None
        json_path_list.append(json_path)

    json_path_list = list(set(json_path_list)) # 去重
    op_param_list = _assemble_opc_params(soc_version, compile_res_dir, op_debug_config, log_level, json_path_list)
    if not op_param_list:
        return None, None
    return json_path_list, op_param_list

def debug_kernel_compile(parameters: OptionParser):
    """
    debug compilation mode
    :param path: 
    :return: True/False
    """
    # 获取执行参数
    op_params_dir, kernel_name, op_debug_config_path, soc_version, package_name, output_path, job, log_level = (
        path_conversion(parameters.full_options["--op_params_dir"].get_value())[-1],
        parameters.full_options["--kernel_name"].get_value(),
        path_conversion(parameters.full_options["--op_debug_config"].get_value())[-1],
        parameters.full_options["--soc_version"].get_value(),
        parameters.full_options["--package_name"].get_value(),
        path_conversion(parameters.full_options["--output_path"].get_value())[-1],
        int(parameters.full_options["--job"].get_value()),
        parameters.full_options["--log"].get_value().lower(),
    )
    short_soc_version = trans_soc_ver_to_short(soc_version)
    parse_op_debug_config_flag, op_debug_config = _parse_op_debug_config(op_debug_config_path)
    if not parse_op_debug_config_flag:
        return False
    debug_compiler_params = namedtuple("debug_compiler_params", \
                                       "op_params_dir, kernel_name, op_debug_config_path, \
                                        short_soc_version, package_name, output_path, log_level")
    _print_debug_compiler_params(debug_compiler_params(op_params_dir, kernel_name, op_debug_config_path, \
                                 short_soc_version, package_name, output_path, log_level))
    
    # 初始化相关路径
    related_path_flag, ascend_opp_path, dyn_binary_path = _init_related_path()
    if not related_path_flag:
        logger.error("failed to initialize ascend_opp_path and dyn_binary_path paths.")
        return False 

    # 初始化所有算子的dyn_binary_info
    init_dyn_binary_info_flag, dyn_binary_info = init_dyn_binary_info(short_soc_version, dyn_binary_path)
    if not init_dyn_binary_info_flag:
        return False  

    try:
        # opc编译
        formatted_time = time_stamp()
        compile_res_dir = f'{output_path}/tmp_res_{formatted_time[:12]}'
        binary_json_file = f'{compile_res_dir}/binary_info_config.json'
        if not make_dirs(compile_res_dir):
            return False

        # 1.组装编译命令参数    
        assemble_compile_params = [op_params_dir, kernel_name, short_soc_version, dyn_binary_info, soc_version, \
                                   compile_res_dir, op_debug_config, log_level]
        json_path_list, op_param_list = assemble_compile_param(assemble_compile_params)
        if not op_param_list:
            return False

        # 2.调用opc工具进行算子编译
        # 2.1 并行编译
        from multiprocessing import Pool
        # 配置编译进程数
        job_num = len(op_param_list) if job >= len(op_param_list) else job
        if job_num < 1:
            logger.error(f'job must greater or equal than 1, but actual is {job_num}, reset to 1')
            job_num = 1
        logger.debug(f'multi jobs num has been set: {job_num}')
        
        pool = Pool(job_num)
        try:
            pool.map(_wrapper_func, op_param_list)
        except Exception as e:
            logger.error(f"process exception, {e}")
        finally:
            pool.close()
            pool.join()

        # 3.检查opc编译结果
        opc_result_flag, real_json_path_list = _check_opc_result(json_path_list, compile_res_dir)
        if not opc_result_flag:
            logger.error(f"OPC compilation result is mistakes.")
            return False

        # 生成binary_info_config
        binary_info_config_flag, binary_info_config_json = \
            gen_debug_binary_info_config(short_soc_version, compile_res_dir, json_path_list)
        if not binary_info_config_flag:
            logger.error(f"failed to gen binary_info_config json.")
            return False
        try:
            with open(binary_json_file, 'w') as file_path:
                json.dump(binary_info_config_json, file_path, sort_keys=True, indent=2, separators=(',', ':'))
        except Exception as e:
            logger.exception(f"failed to write the {binary_json_file} file.")

        # 组装debug.run包相关路径文件
        package_params = namedtuple("package_params", \
                                    "soc_version, package_name, output_path, \
                                    compile_res_dir, binary_json_file, \
                                    binary_info_config_json, dyn_binary_path")
        if not _create_operator_package(package_params(short_soc_version, package_name, output_path, \
                                                       compile_res_dir, binary_json_file, \
                                                       binary_info_config_json, dyn_binary_path)):
            return False

        # 生成debug.run包
        status, data = _create_run_package_command(ascend_opp_path, package_name, output_path)
        if not status:
            logger.error(f'run_package_cmd execute failed. the reason is {data}')
            return False
    
    finally:
        # 删除临时文件
        remove_tempfile([compile_res_dir, os.path.join(output_path, "delivery")])

    logger.info(f"generate run package {package_name} success")
    return True