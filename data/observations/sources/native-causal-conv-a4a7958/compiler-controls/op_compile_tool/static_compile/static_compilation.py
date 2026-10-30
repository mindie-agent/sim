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

import itertools
import json
import copy
import os
import shutil
import subprocess
import sys
import stat
import hashlib
import time
from collections import namedtuple
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from tbe.common.utils import log as logger
from op_compiler_parse_args import OptionParser
from static_compile.gen_binary_info_config import gen_binary_info_config
from op_compiler_common import time_stamp
from op_compiler_common import trans_soc_ver_to_short
from op_compiler_common import make_dirs
from op_compiler_common import remove_tempfile
from op_compiler_common import generate_version_info
from op_compiler_common import get_makeself_file

# 运行时禁止生成__pycache__/文件夹
sys.dont_write_bytecode = True
SCRIPT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

DETERMIN_DIR_MAP = {'false': 'd0', 'true': 'd1'}
# 留多种精度模式可能性
IMPL_MODE_DIR_MAP = {'': ''}

# black list of ops : will not support static compile
OP_BLACK_LIST = {'FlashAttentionScore', 'FlashAttentionScoreGrad'}

_PCIE_THROUGH_ENV = "OP_PCIE_THROUGH_ACCESS_HOST_MEM_CHECK_ENABLE"
_PCIE_THROUGH_SUPPORTED_SOC = ("ascend950")


def _is_pcie_through_globally_enabled(soc_version):
    if not soc_version:
        return False
    try:
        short_soc_version = trans_soc_ver_to_short(soc_version)
    except Exception:
        return False
    if short_soc_version.lower() not in _PCIE_THROUGH_SUPPORTED_SOC:
        return False
    if not os.environ.get(_PCIE_THROUGH_ENV):
        return False
    return True


def _get_extra_option(deter, impl):
    return f'{DETERMIN_DIR_MAP[deter]}{IMPL_MODE_DIR_MAP[impl]}'


def _generate_list(operator_info_list, values, operator_info, operator_type):
    """
    :param operator_info_list: 空列表
    :param values: 当前算子的所有信息
    :param operator_info: 封装算子信息
    :param operator_type: 算子类型
    :return: 封装后的算子信息列表
    """
    for key, staticlist_info in values.items():
        if key == 'staticList':
            for i in staticlist_info:
                operator_info_list.append(operator_info(operator_type, i.get('simplifiedKey'), i.get('binPath')))


def parse_json(output_path):
    """
    :param output_path: 产物输出路径
    :return: 各个算子的信息列表，每个算子为其中一个元素
    """
    json_path = os.path.join(output_path, 'binary_info_config.json')
    with open(json_path, mode='r', encoding='utf-8') as f:
        data = json.load(f)
    operator_info_list = []
    operator_info = namedtuple('operator_info', 'operator_type, simplified_key, bin_path')
    for operator_type, values in data.items():
        _generate_list(operator_info_list, values, operator_info, operator_type)
    logger.debug(f'operator_info_list is {operator_info_list}')
    return operator_info_list


def _generate_dict(tmp_dir):
    """
    :param tmp_dir: 产物生成路径
    :return: 编译产物字典{‘产物名称’: '产物路径'}
    """
    file_dic = {}
    for root, _, files in os.walk(tmp_dir):
        for file in files:
            if file.endswith('.o') or file.endswith('.json'):
                file_dic[file] = os.path.join(root, file)
    return file_dic


def _limit_file_join(file_join, shape_file) -> str:
    if len(file_join) > 256:
        hash_value = hashlib.new('sha256')
        hash_value.update(f'{file_join}'.encode())
        file_sha256 = f'dismiss_too_long_path_{hash_value.hexdigest()}_{shape_file}'
        return file_sha256
    return file_join


def copy_files(soc_version, package_name, output_path, tmp_dir, shape_info_dir):
    """
    :param soc_version: 芯片名代号
    :param package_name: 包名
    :param output_path: 产物输出路径
    :param tmp_dir: 产物生成路径
    :param shape_info_dir: 输入的json文件路径
    :return: True/False
    """
    operator_info_list = parse_json(output_path)
    ascend_opp_path = f'{output_path}/delivery'
    operator_prefix_path = os.path.join(ascend_opp_path, 'operator_package', 'static_kernel', 'ai_core',
                                        package_name[:-4])
    logger.info('====start of copy files====')
    try:
        if os.path.exists(ascend_opp_path):
            shutil.rmtree(ascend_opp_path)
        make_dirs(os.path.join(operator_prefix_path, 'scripts'))
        make_dirs(os.path.join(operator_prefix_path, 'config', soc_version))
        make_dirs(os.path.join(operator_prefix_path, 'config', soc_version, 'shape_info'))
        make_dirs(os.path.join(ascend_opp_path, 'operator_package', 'static_kernel', 'ai_core', 'config'))

        file_dic = _generate_dict(tmp_dir)
        logger.debug(f'file dic: {file_dic}')
        for operator_info in operator_info_list:
            operator_o_name = operator_info.bin_path.split('/')[-1]
            operator_json_name = operator_o_name.replace('.o', '.json')
            dst_path = '/'.join(operator_info.bin_path.split('/')[1:-1])
            config_soc_version = dst_path.split('/')[0]
            if config_soc_version != soc_version:
                logger.error(f'soc_version({config_soc_version}) of binary_info_config.json or '
                              f'script input --soc-version({soc_version}) is not correct')
                return False
            make_dirs(os.path.join(operator_prefix_path, dst_path))
            logger.debug(f'dst_path: {dst_path}')
            if operator_o_name in file_dic:
                shutil.copy(file_dic[operator_o_name],
                            os.path.join(operator_prefix_path, dst_path))
            else:
                logger.error(f'{operator_o_name} is not found, please check.')
                return False
            if operator_json_name in file_dic:
                shutil.copy(file_dic[operator_json_name],
                            os.path.join(operator_prefix_path, dst_path))
            else:
                logger.error(f'{operator_json_name} is not found, please check.')
                return False

        # 拷贝安装卸载脚本/binary_info_config.json
        shutil.copy(os.path.join(f'{SCRIPT_DIR_PATH}', 'install.py'),
                    os.path.join(operator_prefix_path, 'scripts'))
        shutil.copy(os.path.join(f'{SCRIPT_DIR_PATH}', 'uninstall.py'),
                    os.path.join(operator_prefix_path, 'scripts'))
        shutil.copy(os.path.join(f'{SCRIPT_DIR_PATH}', 'uninstall.sh'),
                    os.path.join(operator_prefix_path, 'scripts'))
        shutil.copy(os.path.join(f'{SCRIPT_DIR_PATH}', 'help.info'),
                    os.path.join(operator_prefix_path, 'scripts'))
        shutil.copy(os.path.join(f'{output_path}', 'binary_info_config.json'),
                    os.path.join(operator_prefix_path, 'config', soc_version))

        # 拷贝shape_info_dir目录下的文件
        for root, _, files in os.walk(shape_info_dir):
            for shape_file in files:
                if not shape_file.endswith('.json'):
                    continue
                file_str = os.path.join(root, shape_file)
                file_join = ('_'.join(file_str.split('/'))).strip('_')
                file_join = _limit_file_join(file_join, shape_file)
                shutil.copy(os.path.join(root, shape_file),
                            os.path.join(operator_prefix_path, 'config', soc_version, 'shape_info', file_join))

        # 生成config.ini并拷贝到对应目录
        with open(f'{ascend_opp_path}/operator_package/static_kernel/ai_core/config.ini', mode='wt',
                  encoding='utf-8') as f:
            f.write(f"load_priority={package_name[:-4]}")

        # 生成version.info
        flag, version, version_dir = generate_version_info()
        if flag:
            with open(f'{ascend_opp_path}/operator_package/static_kernel/ai_core/version.info', mode='wt',
                      encoding='utf-8') as file:
                file.write(version)
        else:
            logger.error("failed to obtain the OPP version info")
            return False
    except Exception as e:
        logger.error(e)
        return False
    finally:
        logger.info('====end of copy files====')
    return True


def is_bsdtar_installed():
    return shutil.which("bsdtar") is not None


def compose_makeself_command(ascend_opp_path, package_name, output_path) -> tuple:
    """
    组装makeself包打包命令
    :param ascend_opp_path: 环境变量
    :param package_name: run包名
    :param output_path: run包输出路径
    :return: makeself包打包命令
    """
    makeself_tool, makeself_header = get_makeself_file(ascend_opp_path)
    if not makeself_tool or not makeself_header:
        return None, "makeself_tool or makeself_header does not exist."

    help_info = os.path.join(output_path, 'delivery', 'operator_package', 'static_kernel',
                           'ai_core', package_name[:-4], 'scripts', 'help.info')
    install_script = os.path.join('static_kernel', 'ai_core',
                                package_name[:-4], 'scripts', 'install.py')

    comments = '"STATIC KERNEL RUN PACKAGE"'

    base_commands = [
        makeself_tool,
        "--header", makeself_header,
        "--help-header", help_info,
        '--complevel', '4',
        '--nomd5', '--sha256',
        '--nooverwrite', '--chown',
        '--tar-extra',
        '--numeric-owner',
        '--tar-quietly'
    ]

    if is_bsdtar_installed():
        base_commands.append('--tar-format')
        base_commands.append('pax')
    else:
        base_commands.append('--tar-format')
        base_commands.append('gnu')

    final_commands = itertools.chain(
        base_commands,
        [f'{output_path}/delivery/operator_package',
         package_name,
         comments,
         install_script]
    )

    command = ' '.join(final_commands)
    return command, ''


def create_run_package_command(package_name, output_path):
    """
    :param package_name: run包名
    :param output_path: run包输出路径
    :return: True/False
    """
    ascend_opp_path = os.environ.get('ASCEND_OPP_PATH')
    if not ascend_opp_path:
        return False, 'please setting ASCEND_OPP_PATH'
    makeself_cmd, data = compose_makeself_command(ascend_opp_path,
                                                  package_name,
                                                  output_path
                                                  )
    if not makeself_cmd:
        logger.error(data)
        return False, ''
    run_package_cmd = f'cd {output_path} && {makeself_cmd}'
    logger.debug(f'run_package_cmd: {run_package_cmd}')
    status, data = subprocess.getstatusoutput(run_package_cmd)
    if status != 0:
        return False, data
    return True, data


def rm_run_package(output_path, package_name):
    """
    :param output_path: run包输出路径
    :param package_name: run包名称
    :return:None
    """
    rm_cmd = f'rm -rf {output_path}/{package_name}'
    logger.debug(f'rm_cmd: {rm_cmd}')
    status, data = subprocess.getstatusoutput(rm_cmd)
    if status != 0:
        logger.error(f'remove run package{package_name} failed. the reason is {data}')


def install_run_package(output_path, package_name):
    """
    :param output_path: run包输出路径
    :param package_name: run包名称
    :return: True/False
    """
    install_cmd = f'cd {output_path} && ' \
                  f'chmod +x {package_name} ' \
                  f'&& ./{package_name}'
    logger.debug(f'install_cmd: {install_cmd}')
    status, data = subprocess.getstatusoutput(install_cmd)
    if status != 0:
        logger.error(f'install run package ({package_name}) failed. the reason is {data}')
        return False
    else:
        logger.info(data)
    rm_run_package(output_path, package_name)
    return True


def _modify_fileds_into_opc_format(param):
    dtype = param.get('dtype')
    if dtype == 'float':
        param['dtype'] = 'float32'
    param_type = param.get('param_type')
    if param_type is not None:
        param['paramType'] = param_type
        del param['param_type']
    const_data = param.get('const_data')
    if const_data is not None:
        param['const_value'] = const_data
        del param['const_data']


def _modify_json_file_into_opc_format(params):
    for param in params:
        if param is None:
            continue
        if isinstance(param, list):
            for item in param:
                _modify_fileds_into_opc_format(item)
        else:
            _modify_fileds_into_opc_format(param)


def _modify_json_file_into_opc_for_platform_info(params, json_file, soc_version=None):
    platform_info = params.get("platform_info")
    cube_core_cnt = platform_info.get("cube_core_cnt")
    vector_core_cnt = platform_info.get("vector_core_cnt")
    if cube_core_cnt is None or vector_core_cnt is None:
        raise Exception(f"error: cannot find field cube_core_cnt or vector_core_cnt in dump shape json: {json_file}")
    extra_settings = [
        {
            "key": "_op_aicore_num",
            "value": str(cube_core_cnt),
            "type": "addition"
        },
        {
            "key": "_op_vectorcore_num",
            "value": str(vector_core_cnt),
            "type": "addition"
        },
    ]

    pcie_through_flag = platform_info.get("pcie_through_flag")
    if pcie_through_flag is True and _is_pcie_through_globally_enabled(soc_version):
        extra_settings.append(
            {
                "key": "pcie_through_flag",
                "value": True,
                "type": "addition"
            }
        )

    original_dict = {item["key"]: item for item in params.get("extra_settings", [])}
    new_dict = {item["key"]: item for item in extra_settings}
    original_dict.update(new_dict)
    params["extra_settings"] = list(original_dict.values())

    del params["platform_info"]


def _trans_dump_file_into_opc_format(json_dir, json_file, opc_dump_dir, soc_version=None):
    with open(os.path.join(json_dir, json_file), encoding='utf-8') as file:
        js = json.load(file)
        op_type = js.get('op_type')
        if op_type is None:
            raise Exception(f"error: cannot find field op_type in dump shape json: {json_file}")
        del js['op_type']
        for params in ['inputs', 'outputs', 'attrs']:
            _modify_json_file_into_opc_format(js.get(params))
        bin_filename = js.get('bin_filename')
        if bin_filename is None:
            js['bin_filename'] = json_file[:-5]
        platform_info = js.get("platform_info")
        if platform_info is not None:
            _modify_json_file_into_opc_for_platform_info(js, json_file, soc_version)
        js_out = {}
        js_out['op_type'] = op_type
        js_out['op_list'] = [js]
        wr_flag = os.O_WRONLY | os.O_CREAT
        wr_mode = stat.S_IWUSR | stat.S_IRUSR
        dump_opc_format_path = os.path.join(opc_dump_dir, json_file)
        with open(dump_opc_format_path, 'w') as file_path:
            json.dump(js_out, file_path, sort_keys=True, indent=2, separators=(',', ':'))


def _trans_dump_dir_into_opc_format(shape_info, opc_dump_dir, soc_version=None):
    make_dirs(opc_dump_dir)
    opc_json_list = []
    for root, _, files in os.walk(shape_info):
        for file in files:
            if file.endswith('.json'):
                _trans_dump_file_into_opc_format(root, file, opc_dump_dir, soc_version)
                opc_json_list.append(os.path.join(opc_dump_dir, file))
    return opc_json_list


def _opc_compile_single(\
    opc_json, compile_res_dir, kernel_meta_dir, soc_version, log_level, compile_log_dir, enable_sk):
    aicore_num = 0
    with open(opc_json, encoding='utf-8') as file:
        js = json.load(file)
        js_op = js['op_list'][0]
        deter = js_op.get('deterministic')
        deter_option = ''
        if deter is not None and deter in ['true', 'false']:
            deter_option = f'--deterministic={deter}'
        extra_settings = js_op.get("extra_settings")
        if extra_settings is not None:
            for setting in extra_settings:
                key = setting.get("key", "")
                value = setting.get("value", "")
                if key == "_op_aicore_num":
                    aicore_num = int(value)
                    break
    opc_compile_md = f'opc --input_param={opc_json} '   \
                     f'--output={compile_res_dir} '     \
                     f'--debug_dir={kernel_meta_dir} '  \
                     f'--op_mode=static '               \
                     f'--soc_version={soc_version} '    \
                     f' {deter_option} '                \
                     f'--log={log_level} '              \
                     f'--op_relocatable_kernel_binary={enable_sk}'
    if log_level == 'debug':
        opc_compile_md += ' --op_debug_config=dump_cce'
    if aicore_num != 0:
        opc_compile_md += f' --aicore_num={aicore_num}'
    logger.debug(f'opc_compile_md is: {opc_compile_md}')

    env = os.environ.copy()
    env["ASCEND_SLOG_PRINT_TO_STDOUT"] = "1"

    log_file = os.path.join(compile_log_dir, f'{(os.path.basename(opc_json))[:-5]}_compile_error.log')
    with open(log_file, 'w') as f:
        result = subprocess.run(opc_compile_md,
                                shell=True,
                                check=False,
                                stdout=f,
                                stderr=subprocess.STDOUT,
                                env=env)
    try:
        if result.returncode == 0:
            with open(log_file, 'r+', errors='replace') as f:
                content = f.read()
                f.seek(0, 0)
                f.write(f'op_compiler call opc succeeded, command is : {opc_compile_md}\n')
                f.write(content)
            # 重命名日志文件
            new_log_file = log_file.replace('_compile_error.log', '_compile_succ.log')
            os.rename(log_file, new_log_file)
        else:
            with open(log_file, 'r+', errors='replace') as f:
                content = f.read()
                f.seek(0, 0)
                f.write(f'op_compiler call opc failed, command is : {opc_compile_md}\n')
                f.write(content)
            logger.warn(f'op_compiler call opc execute error, please check: {log_file}')
    except Exception as e:
        logger.warn(f'unexpected error when processing compile log for {opc_json}, '
                     f'error: {e}, please check log: {log_file}')
    return result.returncode


def _summary_from_compile_result_logs(compile_log_dir, opc_json_list_len, log_level):
    log_files = [f for f in os.listdir(compile_log_dir) if f.endswith('_compile_error.log')]
    if log_files:
        summary_file = os.path.join(compile_log_dir, 'summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f'there are {len(log_files)}/{opc_json_list_len} JSON files that failed to static compile.\n')
            for i, log_file in enumerate(log_files, 1):
                json_file = log_file.replace('_compile.log', '.json')
                f.write(f'{i}. {json_file}\n')
        error_log = f'{len(log_files)}/{opc_json_list_len} JSON files compile failed, '\
                    f'please check {summary_file}'
        summary_parent_dir = os.path.basename(compile_log_dir)
        print(f'[summary] {opc_json_list_len - len(log_files)}/{opc_json_list_len} JSON files '
              f'compiled successfully, check {summary_parent_dir}/summary.txt for further details')
        logger.warn(error_log)
    elif log_level != 'debug':
        remove_tempfile([compile_log_dir])
        succ_log = f'{opc_json_list_len}/{opc_json_list_len} JSON files compiled successfully.'
        print(f'[summary] {succ_log}')
        logger.info(f'{succ_log} {compile_log_dir} is empty, delete it.')


def _opc_compile(opc_params):
    if not os.path.isdir(opc_params.shape_info):
        logger.error(f'{opc_params.shape_info} is not existential, please check.')
        return False
    make_dirs(opc_params.compile_res_dir)
    make_dirs(opc_params.kernel_meta_dir)
    os.chmod(opc_params.kernel_meta_dir, 0o750)
    make_dirs(opc_params.compile_log_dir)
    os.chmod(opc_params.compile_log_dir, 0o750)
    opc_json_list = _trans_dump_dir_into_opc_format(opc_params.shape_info, opc_params.opc_dump_dir,
                                                      opc_params.soc_version)

    with Pool(opc_params.job) as pool:
        for _ in tqdm(pool.imap_unordered(
            partial(_opc_compile_single,
                    compile_res_dir=opc_params.compile_res_dir,
                    kernel_meta_dir=opc_params.kernel_meta_dir,
                    soc_version=opc_params.soc_version,
                    log_level=opc_params.log_level,
                    compile_log_dir=opc_params.compile_log_dir,
                    enable_sk=opc_params.enable_sk),
            opc_json_list
        ), total=len(opc_json_list)):
            pass
    _summary_from_compile_result_logs(opc_params.compile_log_dir, len(opc_json_list), opc_params.log_level)
    return True


def _extent_inputs_single_origin(js):
    if js is None:
        return
    ori_shape = js.get('origin_shape')
    ori_format = js.get('origin_format')
    if ori_shape is not None:
        js['ori_shape'] = ori_shape
        del js['origin_shape']
    if ori_format is not None:
        js['ori_format'] = ori_format
        del js['origin_format']


def _extent_inputs_origins(inputs):
    for item in inputs:
        if isinstance(item, (list, tuple)):
            for single in item:
                _extent_inputs_single_origin(single)
        else:
            _extent_inputs_single_origin(item)


def _extent_shape_json_content_single(js, bin_file_name, extra_option, shape_extent_dir, file):
    if bin_file_name is not None:
        js['bin_filename'] = f'{bin_file_name}_{extra_option}'
    if js.get('inputs') is not None:
        _extent_inputs_origins(js['inputs'])
    if js.get('outputs') is not None:
        _extent_inputs_origins(js['outputs'])
    # when we need impl mode, can add js['impl_mode'] = impl here
    with open(os.path.join(shape_extent_dir, f'{file[:-5]}_{extra_option}.json'), 'w') as file_path:
        json.dump(js, file_path, sort_keys=True, indent=2, separators=(',', ':'))


def _extent_shape_json_content(shape_extent_dir, file, js, file_str, follow_op_params_switch, enable_sk, \
    super_kernel_options):
    # filter : only all and static_kenrel will go on static compile
    tune_mode = js.get('tune_mode')
    if tune_mode is not None and tune_mode not in ['all', 'static_kernel']:
        logger.warn(f'{file} tune mode is "{tune_mode}", only support "all" and "static_kernel".')
        return
    # OP BLACK LIST FILTER
    op_type = js.get('op_type')
    if op_type in OP_BLACK_LIST:
        logger.warn(f'op compiler tool do not support static compile of op type: {op_type}')
        return
    bin_file_name = js.get('bin_filename')
    # if enable super kernel, add these options into json
    if enable_sk:
        js['extra_settings'] = [
            {
                "key": "super_kernel_sub_info",
                "value": {
                    "enable_super_kernel": True,
                    "super_kernel_options": super_kernel_options,
                    "super_kernel_sub_loc": "middle"
                },
                "type": "addition"
            },
            {
                "key": "super_kernel_sub_combine",
                "value": True,
                "type": "addition"
            }
        ]
    # just follow json
    if follow_op_params_switch:
        js['origin_path'] = file_str
        deter_f = js.get('deterministic')
        if deter_f is None:
            raise Exception(f'cannot follow op params, no determinstic info found in json {file_str}.')
        elif deter_f not in ["true", "false"]:
            raise Exception(f'cannot follow op params, determinstic is "{deter_f}" in json {file_str}, '
                            f'which should be "true" or "false".')
        extra_option = _get_extra_option(deter_f, '')
        _extent_shape_json_content_single(js, bin_file_name, extra_option, shape_extent_dir, file)
        return
    for deter in DETERMIN_DIR_MAP:
        for impl in IMPL_MODE_DIR_MAP:
            extra_option = _get_extra_option(deter, impl)
            js['deterministic'] = deter
            js['origin_path'] = file_str
            _extent_shape_json_content_single(js, bin_file_name, extra_option, shape_extent_dir, file)


def _extent_shape_json(shape_info, shape_extent_dir, follow_op_params, enable_sk, super_kernel_options):
    follow_op_params_switch = True if follow_op_params == "true" else False
    if os.path.exists(shape_extent_dir):
        shutil.rmtree(shape_extent_dir)
    if not os.path.exists(shape_extent_dir):
        os.mkdir(shape_extent_dir)
    hash_list = {}
    for root, _, files in os.walk(shape_info):
        for file in files:
            if not file.endswith('.json'):
                continue
            file_str = os.path.join(root, file)
            hash_value = ''
            hash_value_with_time = ''
            with open(file_str, "rb") as f:
                hash_value = hashlib.sha256(f.read()).hexdigest()
                if hash_value in hash_list:
                    logger.info(f"Shape config file {file_str} is duplicated with {hash_list[hash_value]}, just skip")
                    continue
                hash_list[hash_value] = file_str
            ticks = time.time()
            hash_value_with_time = hashlib.new('sha256')
            hash_value_with_time.update(f'{hash_value}_{ticks}'.encode())
            with open(file_str, encoding='utf-8') as f:
                js = json.load(f)
                op_type = js.get('op_type')
                if op_type is None:
                    raise Exception(f'cannot find op_type info in json {file_str} ')
                file_sha256 = f'static_kernel_{op_type}_{hash_value_with_time.hexdigest()}_{os.getpid()}.json'
                logger.info(f"kernel name of shape config file {file_str} will be {file_sha256}")
                _extent_shape_json_content(shape_extent_dir, file_sha256, js, file_str, \
                                           follow_op_params_switch, enable_sk, super_kernel_options)


def static_kernel_compile(parameters: OptionParser):
    # 获取执行参数
    shape_info, soc_version, package_name, compile_mode, \
        output_path, follow_op_params, job, log_level, install_flag, enable_sk, super_kernel_options_str = (
        parameters.full_options["--op_params_dir"].get_value(),
        parameters.full_options["--soc_version"].get_value(),
        parameters.full_options["--package_name"].get_value(),
        parameters.full_options["--compile_mode"].get_value(),
        parameters.full_options["--output_path"].get_value(),
        parameters.full_options["--follow_op_params_jit"].get_value(),
        int(parameters.full_options["--job"].get_value()),
        parameters.full_options["--log"].get_value().lower(),
        None,
        bool(parameters.full_options["--enable_super_kernel"].get_value()),
        parameters.full_options["--super_kernel_options"].get_value(),
    )
    if job < 1:
        logger.error(f'job must greater or equal than 1, but actual is {job}, reset to 1')
        job = 1
    logger.debug(f'multi jobs num has been set: {job}')

    super_kernel_options = None
    # Use super_kernel_options directly as string if super kernel is enabled
    if enable_sk:
        super_kernel_options = super_kernel_options_str
        logger.debug(f'super_kernel_options: {super_kernel_options}')

    shape_extent_dir = f'{output_path}/{os.path.basename(os.path.realpath(shape_info))}_extent'
    _extent_shape_json(shape_info, shape_extent_dir, follow_op_params, enable_sk, super_kernel_options)
    binary_json_file = f'{output_path}/binary_info_config.json'
    try:
        if not os.path.exists(output_path):
            logger.error(f'{output_path} is not existential, please check.')
            return False
        if not os.access(output_path, os.W_OK):
            logger.error(f'{output_path} do not have the write permission., please setting.')
            return False

        formatted_time = time_stamp()
        compile_res_dir = f'{output_path}/tmp_{formatted_time[:12]}'
        opc_dump_dir = f'{output_path}/opcdump_{formatted_time[:12]}'
        kernel_meta_dir = f'{output_path}/kernel_meta_static_{formatted_time[:12]}'
        compile_log_dir = f'{output_path}/{package_name[:-4]}_compile_log'

        if compile_mode in ("tune", "aoe"):
            print(f"[WARNING] compile_mode={compile_mode} is being sunset and no longer supported, "
                  f"automatically switching to opc mode for compilation.")
            logger.warn(f"compile_mode={compile_mode} is being sunset, switch to opc mode automatically.")

        # opc mode
        opc_params = namedtuple('opc_params', 'shape_info, opc_dump_dir, soc_version, job, log_level, \
                                compile_res_dir, kernel_meta_dir, compile_log_dir, enable_sk')
        _opc_compile(opc_params(shape_extent_dir, opc_dump_dir, soc_version, job, log_level,
                                compile_res_dir, kernel_meta_dir, compile_log_dir, enable_sk))

        short_soc_version = trans_soc_ver_to_short(soc_version)

        gen_cfg_ret = gen_binary_info_config(package_name[:-4], short_soc_version, shape_extent_dir, compile_res_dir,
                                             binary_json_file, enable_sk)
        if not gen_cfg_ret:
            logger.warn('static compile tool exited successfully, skipping the packaging stage, '
                'due to no binary_info_config was generated')
            return False

        # 拷贝文件
        status = copy_files(short_soc_version, package_name, output_path, compile_res_dir, os.path.realpath(shape_info))
        if not status:
            logger.error('copy_files execute failed.')
            return False
        logger.debug('copy_files execute success')

        # 打包成run包
        status, data = create_run_package_command(package_name, output_path)
        if not status:
            logger.error(f'run_package_cmd execute failed. the reason is {data}')
            return False
        print(f'generate run package {package_name} success')
        logger.info(f'generate run package {package_name} success')

        # 安装run包
        if install_flag:
            status = install_run_package(output_path, package_name)
            if not status:
                return False
            logger.info(f'install run package {package_name} success')
        return True
    finally:
        delivery_dir = os.path.join(output_path, 'delivery')
        if os.path.exists(delivery_dir):
            shutil.rmtree(delivery_dir)
        if log_level != 'debug':
            remove_tempfile([compile_res_dir, kernel_meta_dir, opc_dump_dir, shape_extent_dir, binary_json_file])
