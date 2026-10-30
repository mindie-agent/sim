#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
# ============================================================================
"""
conv2d backprop filter schudule.
"""

from __future__ import absolute_import
from __future__ import print_function

from functools import reduce

from tbe import tvm
from tbe.common import platform as tbe_platform
from tbe.common.platform import platform_info as tbe_platform_info
from tbe.common.utils import log
from tbe.common.utils.errormgr import error_manager_util
from tbe.common.utils.op_util.op_util_cube import CONV2D_BACKPROP_FILTER_OP_NAME
from tbe.dsl.base.operation import get_te_var
from tbe.dsl.compute import cube_util
from tbe.dsl.compute.conv2d_backprop_filter_compute import DynamicConv2dBpFilterParams
import tbe.dsl.static_schedule.conv2d_backprop_filter_schedule_util as conv2dBpFilterUtil
from tbe.dsl.static_schedule.conv_util import ConvBpFilterBinaryDynamic
from tbe.dsl.static_schedule.conv_util import get_inout_dtype
from tbe.dsl.static_schedule.util import align
from tbe.dsl.static_schedule.util import ceil
from tbe.dsl.static_schedule.util import DTYPE_BYTE_MAPPING
from tbe.dsl.static_schedule.util import get_load3d_special_factor
from tbe.dsl.static_schedule.util import parse_tbe_compile_para

CUBE_DIM = 16
CUBE_MUL_SHAPE = 256
OPEN_DOUBLE_BUFFER = 2

# bandwidth overhead
LOAD3D_CONSUME = 27
LOAD2D_CONSUME = 12


class CceConv2dBackpropFilterOp:
    """
    CceConv2dBackpropFilterOp: schedule definition of conv2d_backprop_filter

    Functions
    ----------
    __init__ : initialization

    schedule : schedule definition of conv2d_backprop_filter

    """

    def __init__(self, scope, need_tensorize=True, need_pragma=True):
        """
        initialization

        Parameters:
        ----------
        scope : scope definition

        need_tensorize : whether needs tensorize

        need_pragma : whether needs pragma

        Returns
        -------
        None
        """
        self.scope = scope
        self.need_tensorize = need_tensorize
        self.need_pragma = need_pragma
        self.spec_node_list = []
        self.support_l0c2out = tbe_platform_info.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out")
        self.l1_size = tbe_platform_info.get_soc_spec("L1_SIZE")  # L1 size
        self._corenum = tbe_platform_info.get_soc_spec("CORE_NUM")
        self._l0c_size = tbe_platform_info.get_soc_spec("L0C_SIZE")
        self._l0b_size = tbe_platform_info.get_soc_spec("L0B_SIZE")
        self.c0_size = tbe_platform.C0_SIZE
        self._c04_flag = DynamicConv2dBpFilterParams.dma_c04_flag
        self._strideh_read_flag = DynamicConv2dBpFilterParams.strideh_read_flag
        self.linear_embedding_opti_flag = DynamicConv2dBpFilterParams.linear_embedding_opti_flag
        self.l0b_dma_flag = False
        self.tensor_map = {}
        self.var_map = {}
        self._fmap_in_ub = {}
        self._grads_in_ub = {}
        self.binary_mode = DynamicConv2dBpFilterParams.binary_mode
        self.binary_aub_process = False
        self.binary_bub_process = False
        self.flag_load3d_w_split_case = DynamicConv2dBpFilterParams.flag_load3d_w_split_case
        self.l0c_attach_axis = None
        # reverse direction control number
        self.direction_control = 2
        self.reverse_params = {
            "reverse_status": False,
            "reverse_load": False,
            "axis_unit": None,
            "control_reverse_axis": None,
            "reversed_tensor": [],
            "k_axis_split_dict": {}
        }
        self.c_split_flag = False
        self.dw_trans_flag = False
        self.dw_fixpipe_flag = False
        self.fmap_trans_flag = False
        self.grads_trans_flag = False
        self.sch = None
        self.sch_agent = None
        self.binary_schedule = None
        self.group_dict = None
        self.axis_dict = {}
        self.tiling_para_dict = {}
        self.setfmatrix_dict = {}
        self.fmap_fractal_emit_insn_dict = {}
        self.batch_al1 = 1
        self._op_name = CONV2D_BACKPROP_FILTER_OP_NAME
        self.exponent_base = 2

    def schedule(self, res, spec_node_list, sch_list, dynamic_para=None):
        """
        schedule definition of conv2d_backprop_filter

        Parameters:
        ----------
        res :

        spec_node_list :

        sch_list:

        dynamic_para : A dict of dynamic shape parameters

        Returns
        -------
        None
        """

        self.spec_node_list = spec_node_list

        def _get_c04_fmap_l1(tensor, c04_flag):
            if c04_flag:
                fmap_fractal_before = tensor.op.input_tensors[0]
                fmap_matrix = fmap_fractal_before.op.input_tensors[0]
                return fmap_matrix
            return None

        def _get_previous_ub_list(tensor, ub_list):
            """
            get ub fusion tensor.
            """
            def get(tensor):
                """
                find all tensor
                :return: all tensor
                """

                tensor_list = tensor.op.input_tensors
                for one_tensor in tensor_list:
                    # check which tensor has not been checked
                    ub_list[one_tensor.op.name] = one_tensor
                    get(one_tensor)
            get(tensor)

        def _tiling_shape_check():
            """
            do tiling shape paramters general check

            """
            if self.binary_mode:
                return
            al1_shape = tiling.get("AL1_shape")
            bl1_shape = tiling.get("BL1_shape")
            al0_matrix = tiling.get("AL0_matrix")
            bl0_matrix = tiling.get("BL0_matrix")
            cl0_matrix = tiling.get("CL0_matrix")
            if al1_shape and al1_shape[1] < 1:
                dict_args = {
                    'errCode': "m", 'param_name': "AL1_shape[1]", 'param_value': str(al1_shape[1])
                }
                error_manager_util.raise_runtime_error(dict_args)

            if bl1_shape and bl1_shape[1] < 1:
                dict_args = {}
                dict_args['errCode'] = "E64007"
                dict_args['axis_name'] = "n"
                dict_args['param_name'] = "BL1_shape[1]"
                dict_args['param_value'] = str(bl1_shape[1])
                error_manager_util.raise_runtime_error(dict_args)

            if al0_matrix:
                if al0_matrix[0] != cl0_matrix[1]:
                    dict_args = {}
                    dict_args['errCode'] = "E64008"
                    dict_args['axis_name'] = 'm_axis'
                    dict_args['param_1'] = "AL0_matrix"
                    dict_args['param_2'] = "CL0_matrix"
                    dict_args['value_1'] = str(al0_matrix[0])
                    dict_args['value_2'] = str(cl0_matrix[1])
                    error_manager_util.raise_runtime_error(dict_args)

            if bl0_matrix:
                if bl0_matrix[1] != cl0_matrix[0]:
                    dict_args = {}
                    dict_args['errCode'] = "E64008"
                    dict_args['axis_name'] = 'n_axis'
                    dict_args['param_1'] = "BL0_matrix"
                    dict_args['param_2'] = "CL0_matrix"
                    dict_args['value_1'] = str(bl0_matrix[1])
                    dict_args['value_2'] = str(cl0_matrix[0])
                    error_manager_util.raise_runtime_error(dict_args)

            if al0_matrix and bl0_matrix:
                if al0_matrix[1] != bl0_matrix[0]:
                    dict_args = {}
                    dict_args['errCode'] = "E64008"
                    dict_args['axis_name'] = 'k_axis'
                    dict_args['param_1'] = "AL0_matrix"
                    dict_args['param_2'] = "BL0_matrix"
                    dict_args['value_1'] = str(al0_matrix[1])
                    dict_args['value_2'] = str(bl0_matrix[0])
                    error_manager_util.raise_runtime_error(dict_args)

        def _tiling_buffer_check():
            """
            Do buffer paramters general check

            """
            if self.binary_mode:
                return

            block_cout = tiling.get("block_dim")
            al1_pbuff = tiling.get("manual_pingpong_buffer").get("AL1_pbuffer")
            bl1_pbuff = tiling.get("manual_pingpong_buffer").get("BL1_pbuffer")
            al0_pbuff = tiling.get("manual_pingpong_buffer").get("AL0_pbuffer")
            bl0_pbuff = tiling.get("manual_pingpong_buffer").get("BL0_pbuffer")
            l0c_pbuff = tiling.get("manual_pingpong_buffer").get("CL0_pbuffer")
            cub_pbuff = tiling.get("manual_pingpong_buffer").get("L0C_OUTPUT_pbuffer")
            cl0_matrix = tiling.get("CL0_matrix")
            cub_matrix = tiling.get("L0C_OUTPUT_matrix")

            # blockIdx must be positive int
            dim_i = 0
            for dim_x in block_cout:
                # 4 is d_dim index, d_dim default 0
                low_limit = 1 if dim_i < 4 else 0
                if dim_x < low_limit:
                    dict_args = {}
                    dict_args["errCode"] = "E64004"
                    dict_args["param_name"] = "tiling.block_dim"
                    dict_args["axis_rule"] = "positive int"
                    dict_args["wrong_axis"] = str(dim_i)
                    dict_args["actual_value"] = str(dim_x)
                    error_manager_util.raise_runtime_error(dict_args)
                dim_i = dim_i + 1

            if (not self.support_l0c2out
                and (cl0_matrix[0] % cub_matrix[0] != 0 or cl0_matrix[1] != cub_matrix[1])):
                dict_args = {}
                dict_args['errCode'] = "E64009"
                error_manager_util.raise_runtime_error(dict_args)

            def _gen_dict_args(name, value):
                dict_args = {}
                dict_args["errCode"] = "E64010"
                dict_args["buffer_name"] = name
                dict_args["value"] = str(value)
                return dict_args

            # only support no dbuffer/ dbuffer
            pbuff_check_map = {"AL1_pbuffer": al1_pbuff, "BL1_pbuffer": bl1_pbuff,
                               "AL0_pbuffer": al0_pbuff, "BL0_pbuffer": bl0_pbuff,
                               "L0C_pbuffer": l0c_pbuff, "L0C_OUTPUT_pbuffer": cub_pbuff}
            for pbuff_type, pbuff_val in pbuff_check_map.items():
                if pbuff_val not in (1, 2):
                    dict_args = _gen_dict_args(pbuff_type, pbuff_val)
                    error_manager_util.raise_runtime_error(dict_args)

        def _k_one_core_align_factor(dw_k):
            """
            k_axis is split by nonfactor, lenth of k in one core must aligned with k_one_core_align_factor
            """
            k_al1 = tiling.get("AL1_shape")[0] if tiling.get("AL1_shape") else dw_k * self.c0_size
            k_bl1 = tiling.get("BL1_shape")[0] if tiling.get("BL1_shape") else dw_k * self.c0_size
            k_one_core_align_factor = tvm.max(k_al1, k_bl1)
            return k_one_core_align_factor

        def _atomic_add(sch, res_cc):
            """
            achieve atomic add according to refactor dw_cc

            """
            # in split_w scene reduce_axis is (N, H, W) else is (n, hw)
            batch, real_k = sch[res_cc].op.reduce_axis[0:2]

            self.tiling_para_dict["batch_dim_factor"] = ceil(batch_fmap, block_dim_batch)
            self.tiling_para_dict["batch_dim_factor"] = tvm.max(1, self.tiling_para_dict.get("batch_dim_factor"))
            batch_core, batch_in = sch_agent[res_cc].split(
                batch, factor=self.tiling_para_dict.get("batch_dim_factor"))

            if self.var_map:
                # for dynamic hw, the reduce axis of res_cc dose not cut k0
                if self.flag_load3d_w_split_case == 1:
                    k_axis_ho, k_axis_wo = sch[res_cc].op.reduce_axis[1:]
                    k_1_multicore, ho_single_core = sch_agent[res_cc].split(
                        k_axis_ho, binary_schedule.k_expr_single_core)
                    sch[res_cc].reorder(k_1_multicore, batch_core, batch_in, ho_single_core, k_axis_wo)
                else:
                    hw_single_core_factor = binary_schedule.k_expr_single_core * self.c0_size
                    k_1_multicore, real_k = sch_agent[res_cc].split(real_k, hw_single_core_factor)
                    sch[res_cc].reorder(k_1_multicore, batch_core, batch_in, real_k)
            else:
                k_one_core_align_factor  = _k_one_core_align_factor(dw_k)

                if self.flag_load3d_w_split_case == 1:
                    k_axis_ho, k_axis_wo = sch[res_cc].op.reduce_axis[1:]
                    k_one_core_align_factor = k_one_core_align_factor // (dw_k * self.c0_size)
                    ho_single_core_factor = align(ceil(height_grads, block_dim_k), k_one_core_align_factor)

                    k_1_multicore, ho_single_core = sch_agent[res_cc].split(k_axis_ho, ho_single_core_factor)
                    sch[res_cc].reorder(k_1_multicore, batch_core, batch_in, ho_single_core, k_axis_wo)
                else:
                    hw_single_core_factor = ceil(hw_pad_1 * self.c0_size, block_dim_k)
                    hw_single_core_factor = align(hw_single_core_factor, k_one_core_align_factor) // self.c0_size
                    real_k, k_in = sch_agent[res_cc].split(real_k, self.c0_size)
                    k_1_multicore, real_k = sch_agent[res_cc].split(real_k, hw_single_core_factor)
                    sch[res_cc].reorder(k_1_multicore, batch_core, batch_in, real_k, k_in)
                    if self.batch_al1 > 1:
                        # batch_ddr reduce in ddr, batch_dwcc reduce in l0c
                        batch_ddr, batch_dwcc = sch_agent[res_cc].split(batch_in, self.batch_al1)
                        sch[res_cc].reorder(k_1_multicore, batch_core, batch_ddr, batch_dwcc, real_k, k_in)

            if self.batch_al1 > 1:
                fused_atomic_write = sch[res_cc].fuse(k_1_multicore, batch_core, batch_ddr)
            else:
                fused_atomic_write = sch[res_cc].fuse(k_1_multicore, batch_core)

            # after rfactor op, dw_cc becomes dw_ddr, original dw_ub and dw_ddr
            # will be dropped
            res_ddr = res_cc
            res_cc = sch_agent.rfactor(res_ddr, fused_atomic_write)

            sch[res_cc].set_scope(tbe_platform_info.scope_cc)
            res_ub = None
            if not self.support_l0c2out:
                res_ub = sch.cache_read(res_cc, tbe_platform_info.scope_ubuf, [res_ddr])
            if self.dw_trans_flag:
                sch[res_ddr].set_scope(tbe_platform_info.scope_cc)
            return res_cc, res_ub, res_ddr

        def _full_k_check():
            """
            set flag whether axis K is fully loaded in L0A and L0B
            return:
            -------
            full_k_l0a: 1 or 0,
                        1 means K is fully loaded in L0A
            full_k_l0b: 1 or 0,
                        1 means K is fully loaded in L0B
            """
            if self.binary_mode:
                return 0, 0
            # if k is fully load in BL1 and
            # there is multi load in N1 and N1 in BL1
            # isn't aligned to kernel_height*kernel_width, then align to it
            if (tiling.get("BL1_shape") and
                tiling.get("BL1_shape")[1] * tiling.get("BL0_matrix")[1] % (
                        kernel_height * kernel_width) != 0) and not self._c04_flag:
                tiling.get("BL1_shape")[1] = (align(tiling.get("BL1_shape")[1] * tiling.get("BL0_matrix")[1],
                                              kernel_height * kernel_width) // tiling.get("BL0_matrix")[1])

            # whether axis K is fully loaded in L0A and L0B
            # excluding axis batch
            if not tiling.get("AL0_matrix"):
                full_k_l0a = 1
            elif self.flag_load3d_w_split_case == 1:
                full_k_l0a = (ceil(height_grads, block_dim_k) == 1
                              and tiling.get("AL0_matrix")[1] * self.c0_size >= width_grads)
            else:
                full_k_l0a = tiling.get("AL0_matrix")[1] // ceil(hw_pad_1, block_dim_k)

            if not tiling.get("BL0_matrix"):
                full_k_l0b = 1
            elif self.flag_load3d_w_split_case == 1:
                full_k_l0b = (ceil(height_grads, block_dim_k) == 1
                              and tiling.get("BL0_matrix")[0] * self.c0_size >= width_grads)
            else:
                full_k_l0b = tiling.get("BL0_matrix")[0] // ceil(hw_pad_1, block_dim_k)

            return full_k_l0a, full_k_l0b

        def _compute_tiling_parts():
            """
            compute the parts or the factors of tensors

            """
            if not self.binary_mode:
                # ka and kb may be different,
                # the min value corresponds to one MMAD,
                # the larger one is []
                if tiling.get("AL0_matrix"):  # dw_k equals to ka if L0A needs tiling
                    dw_k = tiling.get("AL0_matrix")[1]
                elif tiling.get("BL0_matrix"):
                    dw_k = tiling.get("BL0_matrix")[0]
                else:
                    # both fully loaded
                    dw_k = hw_pad_1 // block_dim_k
                    if self.flag_load3d_w_split_case == 1:
                        dw_k =  wo_mad_1 * CUBE_DIM // self.c0_size
                dw_k = align(dw_k, 2) if self.flag_load3d_w_split_case and self.c_split_flag else dw_k
                if not tiling.get("AL0_matrix"):  # if grads no tiling in L0A
                    tiling["AL1_shape"] = []  # then no tiling in L1

                # dw_cc is (fmap_channel_1*kernel_height*kernel_width,
                #          grads_channel_1, C0_grads, C0_fmap)
                dw_tiling_factor = [tiling.get("CL0_matrix")[0], tiling.get("CL0_matrix")[1]]
                if self.c_split_flag:
                    dw_tiling_factor = [tiling.get("CL0_matrix")[0] * 2, tiling.get("CL0_matrix")[1]]
                # nparts N, nparts M
                # dw_tiling_nparts only describe the nparts from single core to L0
                dw_tiling_nparts = [ceil(self.tiling_para_dict.get("n_single_core"), dw_tiling_factor[0]),
                                    ceil(ceil(cout_g // CUBE_DIM, dw_tiling_factor[1]), block_dim_cout)]

                # tiling parameters of dw_ub
                dw_ub_tiling_factor = [tiling.get("L0C_OUTPUT_matrix")[0], tiling.get("L0C_OUTPUT_matrix")[1]]
                dw_ub_tiling_nparts = [ceil(dw_tiling_factor[0], dw_ub_tiling_factor[0]),
                                       ceil(dw_tiling_factor[1], dw_ub_tiling_factor[1])]

                if not self.var_map:
                    k_one_core_align_factor = int(_k_one_core_align_factor(dw_k))
                    if self.flag_load3d_w_split_case == 1:
                        k_one_core_align_factor = k_one_core_align_factor // (dw_k * self.c0_size)
                        ho_single_core_factor = ceil(height_grads, block_dim_k)
                        k_single_core_length = align(ho_single_core_factor,
                                                     k_one_core_align_factor) * dw_k * self.c0_size
                    else:
                        hw_single_core_factor = ceil(hw_pad_1 * self.c0_size, block_dim_k)
                        k_single_core_length = align(hw_single_core_factor, k_one_core_align_factor)
                else:
                    k_single_core_length = ceil(hw_pad_1, block_dim_k) * self.c0_size

                # only support loading one batch to L1 at a time for now
                # cout:out->single core(sc)->L1
                grads_l1_tiling_nparts = [1, 1]
                if tiling.get("AL1_shape"):  # if grads needs tiling in L1
                    if len(tiling.get("AL1_shape")) == 1:  # but no C_1 tiling info
                        tiling["AL1_shape"] = tiling.get("AL1_shape") + [1]
                    # nparts K1 in L1, nparts M1 in L1
                    grads_l1_tiling_nparts = [k_single_core_length // tiling.get("AL1_shape")[0],
                                              dw_tiling_nparts[1] // tiling.get("AL1_shape")[1]]

                fmap_l1_tiling_nparts = [1, 1]
                if tiling.get("BL1_shape"):  # if fmap needs tiling in L1
                    if len(tiling.get("BL1_shape")) == 1:  # but no fkk tiling info
                        tiling["BL1_shape"] = tiling.get("BL1_shape") + [1]  # tiling fkk=1
                    # DDR to L1 [nparts K1, nparts N1]
                    fmap_l1_tiling_nparts = [k_single_core_length // tiling.get("BL1_shape")[0],
                                             ceil(dw_tiling_nparts[0], tiling.get("BL1_shape")[1])]
                l1_2_l0_tiling_nparts = [dw_tiling_nparts[0] // fmap_l1_tiling_nparts[1],
                                         dw_tiling_nparts[1] // grads_l1_tiling_nparts[1]]
                self.tiling_para_dict["dw_tiling_factor"] = dw_tiling_factor
                self.tiling_para_dict["dw_tiling_nparts"] = dw_tiling_nparts
                self.tiling_para_dict["dw_ub_tiling_factor"] = dw_ub_tiling_factor
                self.tiling_para_dict["dw_ub_tiling_nparts"] = dw_ub_tiling_nparts
                self.tiling_para_dict["grads_l1_tiling_nparts"] = grads_l1_tiling_nparts
                self.tiling_para_dict["fmap_l1_tiling_nparts"] = fmap_l1_tiling_nparts
                self.tiling_para_dict["l1_2_l0_tiling_nparts"] = l1_2_l0_tiling_nparts
                self.tiling_para_dict["dw_k"] = dw_k
            else:
                self.tiling_para_dict.update(binary_schedule.update_tiling_nparts())

        def _compute_tiling_factors():
            fmap_l1_tiling_factor_k, grads_l1_tiling_factor_k = None, None
            if self.var_map:
                if reduce_split_mode:
                    if tiling.get("AL1_shape"):
                        grads_l1_tiling_factor_k = \
                            tiling.get("AL1_shape")[0] // (dw_k * self.c0_size)
                    if tiling.get("BL1_shape") and tiling.get("AL1_shape"):
                        fmap_l1_tiling_factor_k = \
                            tiling.get("BL1_shape")[0] // tiling.get("AL1_shape")[0]
                else:
                    if tiling.get("BL1_shape"):
                        fmap_l1_tiling_factor_k = \
                            tiling.get("BL1_shape")[0] // (dw_k * self.c0_size)
                    if tiling.get("BL1_shape") and tiling.get("AL1_shape"):
                        grads_l1_tiling_factor_k = \
                            tiling.get("AL1_shape")[0] // tiling.get("BL1_shape")[0]
            return grads_l1_tiling_factor_k, fmap_l1_tiling_factor_k

        def _reduce_split_mode():
            reduce_split_mode = True
            if self.binary_mode:
                reduce_split_mode = (tiling.get("attach_at_flag").get("abkl1_attach_flag") ==
                                     binary_schedule.tiling_utils.get("attach_less"))
            else:
                reduce_split_mode = grads_l1_tiling_nparts[0] > fmap_l1_tiling_nparts[0]
            return reduce_split_mode

        def _load_order_change_flag():
            """
            decide wether to change A and B's load order in L1 and L0 to improve perf

            """
            if not self.support_l0c2out or self.var_map:
                return False
            if grads_l1_tiling_nparts[0] != fmap_l1_tiling_nparts[0]:
                # only change when kAl1 = kBl1
                return False
            if flag_conv1d_case or flag_all_one_case or flag_w_one_case:
                return False
            if tiling.get("manual_pingpong_buffer").get("AL1_pbuffer") == OPEN_DOUBLE_BUFFER and \
                tiling.get("manual_pingpong_buffer").get("BL1_pbuffer") != OPEN_DOUBLE_BUFFER:
                # when AL1 has double buffer, BL1 doesn't, do not change
                return False
            if tiling.get("manual_pingpong_buffer").get("AL0_pbuffer") == OPEN_DOUBLE_BUFFER and \
                tiling.get("manual_pingpong_buffer").get("BL0_pbuffer") != OPEN_DOUBLE_BUFFER:
                # when AL0 has double buffer, BL0 doesn't, do not change
                return False
            a_overhead_opt_flag = bool(tiling.get("special_optimize_flag") % self.exponent_base)
            if a_overhead_opt_flag:
                # confict with overhead_opt
                return False
            return True

        def _l0_attach():
            """
            achieve Al0 and Bl0 compute at loc or ddr

            """
            if self.var_map:
                l0a_attach_mode = (dynamic_l0a_attach == "dw_ddr")
                l0b_attach_mode = (dynamic_l0b_attach == "dw_ddr")
            else:
                l0a_attach_mode = \
                         ((batch_num_sc == 1) and (full_k_in_l0a == 1))
                l0b_attach_mode = \
                         ((batch_num_sc == 1) and (full_k_in_l0b == 1))
            if load_order_change_flag:
                hw_mad_1_mad_at_outer, hw_mad_1_mad_at_inner = sch_agent[dw_cc].split(
                    self.axis_dict.get("hw_mad_1_mad_at"), 1)
            if tiling.get("AL0_matrix"):
                if l0a_attach_mode:
                    # L0A data is more than that L0C needed, attach to dw_ddr
                    sch[self.tensor_map.get("grads_fractal")].compute_at(
                        sch[self.tensor_map.get("dw_ddr")], self.axis_dict.get("c_grads_mad_at"))
                    l0a_attach_scope = self.tensor_map.get("dw_ddr")
                    l0a_attach_axis = self.axis_dict.get("c_grads_mad_at")
                elif load_order_change_flag:
                    sch[self.tensor_map.get("grads_fractal")].compute_at(sch[dw_cc], hw_mad_1_mad_at_inner)
                    l0a_attach_scope = dw_cc
                    l0a_attach_axis = hw_mad_1_mad_at_inner
                else:
                    sch[self.tensor_map.get("grads_fractal")].compute_at(
                        sch[dw_cc], self.axis_dict.get("hw_mad_1_mad_at"))
                    l0a_attach_scope = dw_cc
                    l0a_attach_axis = self.axis_dict.get("hw_mad_1_mad_at")
            else:  # else: fully load, attach to thread_axis
                sch[self.tensor_map.get("grads_fractal")].compute_at(
                    sch[self.tensor_map.get("dw_ddr")], fused_multi_core)
                l0a_attach_scope = self.tensor_map.get("dw_ddr")
                l0a_attach_axis = fused_multi_core

            if tiling.get("BL0_matrix"):
                if l0b_attach_mode:
                    sch[self.tensor_map.get("fmap_fractal")].compute_at(
                        sch[self.tensor_map.get("dw_ddr")], self.axis_dict.get("c_fmap_mad_at"))
                    l0b_attach_scope = self.tensor_map.get("dw_ddr")
                    l0b_attach_axis = self.axis_dict.get("c_fmap_mad_at")
                elif load_order_change_flag:
                    sch[self.tensor_map.get("fmap_fractal")].compute_at(sch[dw_cc], hw_mad_1_mad_at_outer)
                    l0b_attach_scope = dw_cc
                    l0b_attach_axis = hw_mad_1_mad_at_outer
                else:
                    sch[self.tensor_map.get("fmap_fractal")].compute_at(
                        sch[dw_cc], self.axis_dict.get("hw_mad_1_mad_at"))
                    l0b_attach_scope = dw_cc
                    l0b_attach_axis = self.axis_dict.get("hw_mad_1_mad_at")
            else:  # else: fully load, attach to thread_axis
                sch[self.tensor_map.get("fmap_fractal")].compute_at(
                    sch[self.tensor_map.get("dw_ddr")], fused_multi_core)
                l0b_attach_scope = self.tensor_map.get("dw_ddr")
                l0b_attach_axis = fused_multi_core

            return [l0a_attach_scope, l0a_attach_axis, l0b_attach_scope, l0b_attach_axis]

        def _check_l1_full_load(tensor_name):
            """
            check L1 full load
            """
            if tensor_name == "aL1":
                attach_flag = "al1_attach_flag"
                tiling_shape = "AL1_shape"
            else:
                attach_flag = "bl1_attach_flag"
                tiling_shape = "BL1_shape"
            is_l1_full_load = not tiling.get(tiling_shape)
            if self.binary_mode:
                attach_full_load = binary_schedule.tiling_utils.get("attach_full_load")
                is_l1_full_load = tiling.get("attach_at_flag").get(attach_flag) == attach_full_load
            return (is_l1_full_load)

        def _al1_attach():
            """
            achieve Al1 compute at l0c or ddr

            """
            if self.var_map:
                al1_attach_mode = (dynamic_al1_attach == "dw_cc")
            elif self.flag_load3d_w_split_case == 1:
                al1_attach_mode = (grads_l1_tiling_nparts[0] != 1 or batch_num_sc != 1
                                   or tiling.get("AL0_matrix")[1] * self.c0_size < width_grads)
            else:
                al1_attach_mode = (grads_l1_tiling_nparts[0] != 1 or batch_num_sc != self.batch_al1)
            if reorder_l1_mn:
                run_once_n_dim = [self.axis_dict.get("c_fmap_l1_c1"),
                                  self.axis_dict.get("c_fmap_l1_c1"),
                                  self.axis_dict.get("c_fmap_l1_at")] + c_fmap_mad_at_list
            else:
                run_once_n_dim = c_fmap_mad_at_list
            del_n0_outer_flag = l0a_attach_axis == self.axis_dict.get("c_grads_mad_at") and reorder_flag

            def _grad_matrix_attach(run_once_n_dim):
                a_overhead_opt_flag = bool(tiling.get("special_optimize_flag") % self.exponent_base)
                if not _check_l1_full_load("aL1"):
                    # if axis K(reduce) needs split, then attach to dw_cc
                    if al1_attach_mode:
                        al1_attach_axis = self.axis_dict.get("al1_at_axis")
                        al1_attach_scope = dw_cc
                        if a_overhead_opt_flag:
                            sch[self.tensor_map.get("grads_matrix")].allocate_at(
                                sch[al1_attach_scope], self.axis_dict.get("al1_at_axis"))
                            al1_attach_axis = l0a_attach_axis
                    else:  # if axis K fully load in L1, attach to dw_ddr
                        if self.batch_al1 > 1:
                            al1_attach_axis = self.axis_dict.get("c_grads_multi_l1_at")
                        else:
                            al1_attach_axis = self.axis_dict.get("c_grads_l1_at")
                        al1_attach_scope = self.tensor_map.get("dw_ddr")
                        if a_overhead_opt_flag:
                            # the list of axis is c_grads_mad_at, c_fmap_mad_at
                            run_once_n_dim_tmp = list(set(run_once_n_dim) - set(c_fmap_mad_at_list)) \
                                if del_n0_outer_flag else run_once_n_dim
                            sch[self.tensor_map.get("grads_matrix")].allocate_at(
                                sch[self.tensor_map.get("dw_ddr")], self.axis_dict.get("c_grads_l1_at"),
                                run_once_axes=run_once_n_dim_tmp)
                            al1_attach_scope = l0a_attach_scope
                            al1_attach_axis = l0a_attach_axis
                else:  # else: fully load, attach to thread_axis
                    al1_attach_axis = fused_multi_core
                    al1_attach_scope = self.tensor_map.get("dw_ddr")
                    if a_overhead_opt_flag and tiling.get("AL0_matrix"):
                        if del_n0_outer_flag:
                            # the list of axis is c_grads_mad_at, c_fmap_mad_at
                            run_once_n_dim = list(set(run_once_n_dim) - set(c_fmap_mad_at_list))
                        sch[self.tensor_map.get("grads_matrix")].allocate_at(
                            sch[self.tensor_map.get("dw_ddr")], fused_multi_core, run_once_axes=run_once_n_dim)
                        al1_attach_scope = l0a_attach_scope
                        al1_attach_axis = l0a_attach_axis
                sch[self.tensor_map.get("grads_matrix")].compute_at(sch[al1_attach_scope], al1_attach_axis)
                if self.grads_trans_flag:
                    sch[self.tensor_map.get("grads")].compute_at(sch[al1_attach_scope], al1_attach_axis)
                return al1_attach_scope, al1_attach_axis
            return _grad_matrix_attach(run_once_n_dim)

        def _bl1_attach():
            """
            achieve Bl1 compute at l0c or ddr

            """

            fmap_matrix_flag = not self.var_map or flag_all_one_case
            if self.var_map:
                bl1_attach_mode = (dynamic_bl1_attach == "dw_cc")
            elif self.flag_load3d_w_split_case == 1:
                bl1_attach_mode = (fmap_l1_tiling_nparts[0] != 1 or batch_num_sc != 1
                                   or tiling.get("BL0_matrix")[0] * self.c0_size < width_grads)
            else:
                bl1_attach_mode = (fmap_l1_tiling_nparts[0] != 1 or batch_num_sc != 1)
            run_once_mdim = [self.axis_dict.get("c_grads_mad_at"), self.axis_dict.get("c_grads_l1_at")]
            if reorder_l1_mn:
                run_once_mdim = [self.axis_dict.get("c_grads_mad_at"), ]

            def _fmap_l1_attach(run_once_mdim):
                b_overhead_opt_flag = bool(tiling.get("special_optimize_flag") //
                                           self.exponent_base % self.exponent_base)
                if not _check_l1_full_load("bL1"):
                    # if axis K needs split, then attach to dw_cc
                    if bl1_attach_mode:
                        bl1_attach_axis = self.axis_dict.get("bl1_at_axis")
                        bl1_attach_scope = dw_cc
                        if not flag_all_one_case:
                            sch[self.tensor_map.get("fmap_l1")].compute_at(sch[bl1_attach_scope], bl1_attach_axis)
                    else:  # if axis K fully load in L1, attach to dw_ddr
                        bl1_attach_axis = self.axis_dict.get("c_fmap_l1_at")
                        bl1_attach_scope = self.tensor_map.get("dw_ddr")
                        if not flag_all_one_case:
                            if b_overhead_opt_flag:
                                if not reorder_flag:
                                    # the list of axis is c_fmap_mad_at, c_grads_mad_at
                                    run_once_mdim = list(set(run_once_mdim) - {self.axis_dict.get("c_grads_mad_at")})
                                sch[self.tensor_map.get("fmap_l1")].allocate_at(
                                    sch[self.tensor_map.get("dw_ddr")],
                                    self.axis_dict.get("c_fmap_l1_at"),
                                    run_once_axes=run_once_mdim + run_once_ndim)
                                bl1_attach_scope = self.tensor_map.get("dw_ddr")
                                bl1_attach_axis = self.axis_dict.get("c_fmap_mad_at")
                            if self.l0b_dma_flag:
                                bl1_attach_axis = self.axis_dict.get("c_fmap_mad_at")
                            sch[self.tensor_map.get("fmap_l1")].compute_at(sch[bl1_attach_scope], bl1_attach_axis)
                else:  # else: fully load, attach to thread_axis
                    bl1_attach_axis = fused_multi_core
                    bl1_attach_scope = self.tensor_map.get("dw_ddr")
                    if not flag_all_one_case:
                        if b_overhead_opt_flag and tiling.get("AL0_matrix"):
                            if not reorder_flag:
                                # the list of axis is c_fmap_mad_at, c_grads_mad_at
                                run_once_mdim = list(set(run_once_mdim) - {self.axis_dict.get("c_grads_mad_at"), })
                            sch[self.tensor_map.get("fmap_l1")].allocate_at(
                                sch[self.tensor_map.get("dw_ddr")], fused_multi_core,
                                run_once_axes=run_once_mdim + run_once_ndim)
                            bl1_attach_scope = self.tensor_map.get("dw_ddr")
                            bl1_attach_axis = self.axis_dict.get("c_fmap_mad_at")
                        sch[self.tensor_map.get("fmap_l1")].compute_at(sch[bl1_attach_scope], bl1_attach_axis)

                return bl1_attach_scope, bl1_attach_axis

            bl1_attach_scope, bl1_attach_axis = _fmap_l1_attach(run_once_mdim)

            if self.flag_load3d_w_split_case == 1 and not self.var_map:
                # It is required in current scheme that row-major tensor be attached at L0 scope
                sch[self.tensor_map.get("fmap_matrix")].compute_at(sch[l0b_attach_scope], l0b_attach_axis)
            elif fmap_matrix_flag and not self.l0b_dma_flag:
                sch[self.tensor_map.get("fmap_matrix")].compute_at(sch[bl1_attach_scope], bl1_attach_axis)
            # fmap_l1 axes:
            #   group, batch, hw, fkk(cin_1*hw*wk), mad->16, cin_0->16
            if self.tensor_map.get("fmap_ub") is not None:
                sch[self.tensor_map.get("fmap_ub")].compute_at(
                    sch[self.tensor_map.get("fmap_l1")], sch[self.tensor_map.get("fmap_l1")].op.axis[4])
            return bl1_attach_scope, bl1_attach_axis

        def _double_buffer():
            """
            achieve double_buffer

            """

            def _binary_const_double_buffer(binary_tiling_data):
                if binary_tiling_data.get("al1_pb") == 1:
                    sch[self.tensor_map.get("grads_matrix")].double_buffer()
                    if self.grads_trans_flag:
                        sch[self.tensor_map.get("grads")].double_buffer()
                if binary_tiling_data.get("bl1_pb") == 1:
                    if not flag_all_one_case:
                        sch[self.tensor_map.get("fmap_l1")].double_buffer()
                    else:
                        sch[self.tensor_map.get("fmap_matrix")].double_buffer()
                if binary_tiling_data.get("l0c_pb") == 1:
                    sch[dw_cc].double_buffer()

            def _dynamic_double_buffer():
                cache_tiling = self.binary_schedule.cache_tiling
                sch[self.tensor_map.get("grads_matrix")].double_buffer(cache_tiling.get("al1_pb"))
                if self.grads_trans_flag:
                    sch[self.tensor_map.get("grads")].double_buffer(cache_tiling.get("al1_pb"))
                if not flag_all_one_case:
                    sch[self.tensor_map.get("fmap_l1")].double_buffer(cache_tiling.get("bl1_pb"))
                else:
                    sch[self.tensor_map.get("fmap_matrix")].double_buffer(cache_tiling.get("bl1_pb"))
                sch[dw_cc].double_buffer(cache_tiling.get("l0c_pb"))

            def _static_double_buffer():
                if tiling.get("manual_pingpong_buffer").get("AL1_pbuffer") == OPEN_DOUBLE_BUFFER:
                    sch[self.tensor_map.get("grads_matrix")].double_buffer()
                    if self.grads_trans_flag:
                        sch[self.tensor_map.get("grads")].double_buffer()
                    if tiling.get("manual_pingpong_buffer").get("BL1_pbuffer") == OPEN_DOUBLE_BUFFER:
                        if not flag_all_one_case:
                            sch[self.tensor_map.get("fmap_l1")].double_buffer()
                        else:
                            sch[self.tensor_map.get("fmap_matrix")].double_buffer()
                    if tiling.get("manual_pingpong_buffer").get("CL0_pbuffer") == OPEN_DOUBLE_BUFFER:
                        sch[dw_cc].double_buffer()

            if not conv2dBpFilterUtil.DEBUG_DOUBLE_BUFFER_OFF:
                # binary constantization
                binary_tiling_data = tiling.get("binary_tiling_data")
                if binary_tiling_data:
                    _binary_const_double_buffer(binary_tiling_data)
                # dynamic
                elif self.var_map:
                    _dynamic_double_buffer()
                # static
                else:
                    _static_double_buffer()

                if tiling.get("manual_pingpong_buffer").get("AL0_pbuffer") == OPEN_DOUBLE_BUFFER:
                    sch[self.tensor_map.get("grads_fractal")].double_buffer()

                if tiling.get("manual_pingpong_buffer").get("BL0_pbuffer") == OPEN_DOUBLE_BUFFER:
                    sch[self.tensor_map.get("fmap_fractal")].double_buffer()

                if tiling.get("manual_pingpong_buffer").get("L0C_OUTPUT_pbuffer") == OPEN_DOUBLE_BUFFER:
                    if not self.support_l0c2out:
                        sch[dw_ub].double_buffer()
                _enable_preload()
                _do_double_buffer_aub_bub()

        def _enable_preload():
            """
            do L1 preload with double buffer in static scene in milan
            """

            if self.var_map or (not self.support_l0c2out) or flag_all_one_case:
                return
            # allocate_at and db conflict, AL1/BL1 preload should be in same axis
            if (tiling.get("manual_pingpong_buffer").get("AL1_pbuffer") != OPEN_DOUBLE_BUFFER or
                tiling.get("manual_pingpong_buffer").get("BL1_pbuffer") != OPEN_DOUBLE_BUFFER):
                return
            if (grads_l1_tiling_nparts[0] != fmap_l1_tiling_nparts[0] or
                (grads_l1_tiling_nparts[1] != 1 and fmap_l1_tiling_nparts[1] != 1)):
                return
            sch[self.tensor_map.get("grads_matrix")].preload()
            sch[self.tensor_map.get("fmap_l1")].preload()

        def _do_double_buffer_aub_bub():
            """
            do double buffer for front l1 tensor
            """
            if not self.binary_aub_process:
                return
            if tiling.get("manual_pingpong_buffer").get("AUB_pbuffer") == OPEN_DOUBLE_BUFFER:
                for tensor_name in binary_schedule.ub_tensor_list:
                    tensor = self._grads_in_ub.get(tensor_name)
                    sch[tensor].double_buffer()
            if tiling.get("manual_pingpong_buffer").get("BUB_pbuffer") == OPEN_DOUBLE_BUFFER:
                for tensor_name in binary_schedule.ub_tensor_list:
                    tensor = self._fmap_in_ub.get(tensor_name)
                    sch[tensor].double_buffer()

        def _cal_unit_flag_condition():
            """
            cal binary unit flag condition
            """
            cache_tiling = self.binary_schedule.cache_tiling
            batch_single_core = cache_tiling.get("batch_single_core")
            max_kl1_div_min_kl1 = cache_tiling.get("kl1_times")
            if reduce_split_mode:
                min_kl1_div_kl0 = cache_tiling.get("kal0_factor")
            else:
                min_kl1_div_kl0 = cache_tiling.get("kbl0_factor")
            last_k_idx = self.binary_schedule.k_expr_single_core // cache_tiling.get("k_l0")
            cur_k_idx = 0
            extend_list = [1, min_kl1_div_kl0, min_kl1_div_kl0 * max_kl1_div_min_kl1]
            reduce_axis_dict_batch = sch_agent[dw_cc].get_relate_scope(
                dw_cc.op.reduce_axis[0], self.axis_dict.get("batch_insn")
            )
            reduce_axis_dict_k = sch_agent[dw_cc].get_relate_scope(
                dw_cc.op.reduce_axis[1], self.axis_dict.get("batch_insn")
            )
            for i, k_axis in enumerate(reduce_axis_dict_k[::-1]):
                cur_k_idx += k_axis * extend_list[i]
            condition_k_singlecore = cur_k_idx >= last_k_idx - 1
            condition_batch_singlecore = reduce_axis_dict_batch[0] >= batch_single_core - 1
            condition_k_muticore, condition_batch_muticore = \
                _cal_unit_flag_condition_muticore(cur_k_idx, reduce_axis_dict_batch[0])
            unit_flag_condition = tvm.all(
                tvm.any(condition_k_singlecore, condition_k_muticore),
                tvm.any(condition_batch_singlecore, condition_batch_muticore)
            )
            return unit_flag_condition

        def _cal_unit_flag_condition_muticore(cur_k_idx, cur_batch_idx):
            """
            cal binary unit flag condition in muticore non-factor situation
            """
            cache_tiling = self.binary_schedule.cache_tiling
            batch_single_core = cache_tiling.get("batch_single_core")
            total_n = cin1_g * kernel_height * kernel_width
            total_m = cout_g // CUBE_DIM
            n_l0 = cache_tiling.get("n_ub_l0_time") * cache_tiling.get("cub_n1")
            m_l0 = cache_tiling.get("m_l0")
            if self.c_split_flag:
                # for fp32 scene, n_l0 * n_single_core(even number) is equal to n_l0_n016 * n_single_core_n016
                # because optiling use cin1_g is equal to cing / 8 in cache tiling
                total_n = cube_util.shape_to_list(self.tensor_map.get("dw_ddr").shape)[1]
            # compute real dim for muticore non-factor situation
            real_n_dim = ceil(total_n, cache_tiling.get("n_single_core") * cache_tiling.get("n_bl1") * n_l0)
            real_m_dim = ceil(total_m, cache_tiling.get("m_single_core") * cache_tiling.get("m_al1") * m_l0)
            real_batch_dim = ceil(batch_fmap, ceil(batch_fmap, block_dim_batch))
            block_div_batch = real_n_dim * real_m_dim * block_dim_group
            block_div_k = real_batch_dim * real_n_dim * real_m_dim * block_dim_group
            k_block_offset = block_idx // block_div_k
            batch_block_offset = (block_idx // block_div_batch) % real_batch_dim
            condition_k = tvm.all(
                hw_pad_1 % block_dim_k != 0,  # using simple conditon to reduce scalar compute
                k_block_offset * ceil(hw_pad_1, block_dim_k) + cur_k_idx * dw_k >= hw_pad_1 - dw_k
            )
            condition_batch = tvm.all(
                batch_fmap % block_dim_batch != 0,
                batch_block_offset * batch_single_core + cur_batch_idx >= batch_fmap - 1
            )
            return condition_k, condition_batch

        def _unit_flag_binary():
            """
            enable unit flag, pass the unit flag condition
            """
            is_enable_unit_flag = self.var_map and self.support_l0c2out and not self.flag_load3d_w_split_case
            if is_enable_unit_flag:
                unit_flag_condition = _cal_unit_flag_condition()
                sch[dw_cc].pragma(self.axis_dict.get("batch_insn"), "unit_flag_condition", unit_flag_condition)

        def _emit_insn():
            """
            achieve emit_insn

            """

            mad_dict = _get_matrix_mad_dict()
            self._emit_insn_grads(flag_w_one_case, in_dtype)
            self._emit_insn_fmap(flag_all_one_case, kernel_height, kernel_width, in_dtype)
            # move dw from L0C to UB
            if not self.support_l0c2out:
                sch[dw_ub].emit_insn(dw_ub.op.axis[0], 'dma_copy')

            sch[dw_cc].emit_insn(self.axis_dict.get("batch_insn"), 'mad', mad_dict)
            # in order to enable uf scalar perf optimization, uf pragma should be under mad axis
            # so we do uf pragma after emit_insn
            _unit_flag_binary()

            # manage fixpipe memory
            for fixpipe_l1_mem in self.tensor_map.get("fixpipe_l1", []):
                sch[fixpipe_l1_mem].emit_insn(fixpipe_l1_mem.op.axis[0], "dma_copy")
            if self.tensor_map.get("fixpipe_l1_eltwise") is not None:
                fixpipe_l1_eltwise = self.tensor_map.get("fixpipe_l1_eltwise")
                sch[fixpipe_l1_eltwise].emit_insn(fixpipe_l1_eltwise.op.axis[0], "dma_copy")
            for fixpipe_fb_mem in self.tensor_map.get("fixpipe_fb", {}).values():
                sch[fixpipe_fb_mem].emit_insn(fixpipe_fb_mem.op.axis[0], "dma_copy")

            # move dw from UB to ddr
            dw_ddr_emit_axis = self.axis_dict.get("c_grads_mad_insn") if self.dw_trans_flag else c_fmap_2_ub_insn
            if self.tensor_map.get("dw_ddr").op.tag == "fixpipe_reform":
                sch[self.tensor_map.get("dw_ddr")].emit_insn(dw_ddr_emit_axis, "fixpipe_op")
            else:
                if self.binary_mode == cube_util.BinaryMode.NC1HWC0:
                    sch[self.tensor_map.get("dw_ddr")].pragma(dw_ddr_emit_axis, "loop_with_no_overlap_tensor")
                sch[self.tensor_map.get("dw_ddr")].emit_insn(dw_ddr_emit_axis, 'dma_copy',
                                                             self.binary_schedule.dma_insn_dict)
            sch_list.append(self.tensor_map.get("dw_ddr"))

            _emit_insn_aub_bub()

        def _calc_load3d_channel_size():
            """
            Calculate the channel value of load3d
            """
            # in fp32 scene, c1 in L1 aligned to 16, so we use cin1_g(aligned to 16) here
            conv_fm_c = real_g * cin1_g * c0_fmap if self.c_split_flag else featuremap_channel
            conv_fm_c1 = real_g * cin1_g if self.c_split_flag else c1_fmap
            # for dynamic scense, load3d channel_size should be actual channel
            if self.var_map:
                cache_tiling = self.binary_schedule.cache_tiling
                n_l0 = cache_tiling.get("n_ub_l0_time") * cache_tiling.get("cub_n1")
                n_l1 = cache_tiling.get("n_bl1") * n_l0
                n_l1 = n_l1 * 2 if self.c_split_flag else n_l1  # for fp32, n_l1 should be double
                kernel_hw = kernel_width * kernel_height
                c1 = ceil(n_l1, kernel_hw)
                # if divisible, there is no need to overload
                # if doubling is divisible, load three C1's twice
                # in other cases, the worst scenario is to load 3 C1 at once
                extern_c1 = tvm.select(kernel_hw > n_l1,
                                       tvm.select(kernel_hw % n_l1 != 0, 1, 0),
                                       tvm.select(n_l1 % kernel_hw == 0, 0,
                                                  tvm.select(n_l1 * 2 %
                                                             kernel_hw == 0, 1, 2)
                                                  )
                                       )
                conv_fm_c1 = c1 + extern_c1
                conv_fm_c = conv_fm_c1 * c0_fmap
            return conv_fm_c1, conv_fm_c

        def _get_fmap_and_stride():
            """
            If it is a stride_read scene, update the stride_h and fmap_h
            """
            fmap_h, stride_h = featuremap_height, stride_height
            if self._strideh_read_flag:
                fmh_idx = 3 if self.var_map else 2
                fmap_h = cube_util.shape_to_list(self.tensor_map.get("fmap_l1").shape)[fmh_idx]
                stride_h = kernel_height
            return fmap_h, stride_h

        def _set_fmap_fmatrix_dict():
            """
            Set parameters for fmap matrix
            """
            fmap_h, stride_h = _get_fmap_and_stride()
            conv_fm_c1, conv_fm_c = _calc_load3d_channel_size()
            self.setfmatrix_dict["conv_kernel_h"] = kernel_height
            self.setfmatrix_dict["conv_kernel_w"] = kernel_width
            self.setfmatrix_dict["conv_padding_top"] = pad_up
            self.setfmatrix_dict["conv_padding_bottom"] = pad_down
            self.setfmatrix_dict["conv_padding_left"] = pad_left
            self.setfmatrix_dict["conv_padding_right"] = pad_right
            self.setfmatrix_dict["conv_stride_h"] = stride_h
            self.setfmatrix_dict["conv_stride_w"] = stride_width
            self.setfmatrix_dict["conv_fm_c"] = conv_fm_c
            self.setfmatrix_dict["conv_fm_h"] = fmap_h
            self.setfmatrix_dict["conv_fm_w"] = featuremap_width
            self.setfmatrix_dict["conv_dilation_h"] = dilation_height
            self.setfmatrix_dict["conv_dilation_w"] = dilation_width
            if self.var_map:
                self.setfmatrix_dict["set_fmatrix"] = 1
                self.setfmatrix_dict["conv_fm_c1"] = conv_fm_c1
                self.setfmatrix_dict["conv_fm_c0"] = c0_fmap
            if self.flag_load3d_w_split_case == 1:
                self.setfmatrix_dict["conv_w_split"] = True

        def _set_fmap_fractal_dict():
            """
            Set parameters for fmap fractal
            """
            fmap_h, stride_h = _get_fmap_and_stride()
            conv_fm_c1, conv_fm_c = _calc_load3d_channel_size()
            self.fmap_fractal_emit_insn_dict["conv_kernel_h"] = kernel_height
            self.fmap_fractal_emit_insn_dict["conv_kernel_w"] = kernel_width
            self.fmap_fractal_emit_insn_dict["conv_padding_top"] = pad_up
            self.fmap_fractal_emit_insn_dict["conv_padding_bottom"] = pad_down
            self.fmap_fractal_emit_insn_dict["conv_padding_left"] = pad_left
            self.fmap_fractal_emit_insn_dict["conv_padding_right"] = pad_right
            self.fmap_fractal_emit_insn_dict["conv_stride_h"] = stride_h
            self.fmap_fractal_emit_insn_dict["conv_stride_w"] = stride_width
            self.fmap_fractal_emit_insn_dict["conv_fm_c"] = conv_fm_c
            self.fmap_fractal_emit_insn_dict["conv_fm_h"] = fmap_h
            self.fmap_fractal_emit_insn_dict["conv_fm_w"] = featuremap_width
            self.fmap_fractal_emit_insn_dict["group_flag"] = 1
            self.fmap_fractal_emit_insn_dict["l1_group_flag"] = 1

            self.fmap_fractal_emit_insn_dict["conv_dilation_h"] = dilation_height
            self.fmap_fractal_emit_insn_dict["conv_dilation_w"] = dilation_width
            if self.flag_load3d_w_split_case == 1:
                self.fmap_fractal_emit_insn_dict["conv_w_split"] = True
            if self.var_map:
                self.fmap_fractal_emit_insn_dict["set_fmatrix"] = 0
                self.fmap_fractal_emit_insn_dict["conv_fm_c1"] = conv_fm_c1
                self.fmap_fractal_emit_insn_dict["conv_fm_c0"] = c0_fmap
            if self.binary_mode:
                # The following conditions for the N direction are added to the emit_insn of im2col:
                # realg_idx * cin1_g + cin1_idx < cin_ori
                # to prevent no set_fmatrix but im2col is performed
                self.fmap_fractal_emit_insn_dict["real_fm_c1_ext"] = cin1_g
                self.fmap_fractal_emit_insn_dict["real_fm_c1"] = c1_fmap
                self.fmap_fractal_emit_insn_dict["enable_c1_restrict"] = True

        def _get_matrix_mad_dict():
            """
            get_set_matrix_dict
            """
            mad_dict = {"mad_pattern": 3,
                        "k_outer": _get_k_outer_info(),
                        "enable_k_alignment": int(self.c_split_flag)}

            if not self.var_map:
                mad_dict["mad_pattern"] = 2
            if self.flag_load3d_w_split_case == 1:
                mad_dict["k_outer"] = [
                    self.axis_dict.get("batch_insn_o"), self.axis_dict.get("wo_l0c_multi"),
                    self.axis_dict.get("hw_mad_1_l1_out_at"), self.axis_dict.get("hw_mad_1_l1_in_at"),
                    self.axis_dict.get("hw_mad_1_mad_at")
                ]

            if self.binary_mode and self.c_split_flag:
                mad_dict["hf32"] = get_te_var("hf32_flag").get_tvm_var()
            elif tiling_dtype[conv2dBpFilterUtil.MAD_DTYPE_INDEX] == "hfloat32":
                mad_dict["hf32"] = 1

            if (tbe_platform.intrinsic_check_support("Intrinsic_vconv", "bf162f32")
                    and get_te_var("is_bf16") is not None):
                mad_dict["datatype_bf16"] = get_te_var("is_bf16").get_tvm_var()

            _set_fmap_fmatrix_dict()
            _set_fmap_fractal_dict()
            return mad_dict

        def _get_k_outer_info():
            """
            if dw_cc is revsersed pragma will be reversed too
            """
            if dw_cc in self.reverse_params.get("reversed_tensor"):
                control_reverse_axis = self.reverse_params.get("control_reverse_axis")
                axis_unit = self.reverse_params.get("axis_unit")
                k_outer_info = [
                    self.axis_dict.get("batch_insn_o"), self.axis_dict.get("hw_mad_1_l1_out_at"),
                    self.axis_dict.get("hw_mad_1_l1_in_at"), self.axis_dict.get("hw_mad_1_mad_at")]
                k_axis_split_dict = self.reverse_params.get("k_axis_split_dict")

                k_outer_info_reverse = []
                for k_axis in k_outer_info:
                    if k_axis in k_axis_split_dict:
                        new_k_outer, new_k_inner = k_axis_split_dict.get(k_axis)
                        k_outer_info_reverse.append(
                            tvm.select(control_reverse_axis % self.direction_control == 0,
                                       axis_unit.get(new_k_outer)[-1] - 1 - new_k_outer.var, new_k_outer))
                        k_outer_info_reverse.append(new_k_inner)
                    else:
                        k_outer_info_reverse.append(
                            tvm.select(control_reverse_axis % self.direction_control == 0,
                                       axis_unit.get(k_axis)[-1] - 1 - k_axis.var, k_axis))
                return k_outer_info_reverse
            else:
                k_outer_info = [
                    self.axis_dict.get("batch_insn_o"), self.axis_dict.get("hw_mad_1_l1_out_at"),
                    self.axis_dict.get("hw_mad_1_l1_in_at"), self.axis_dict.get("hw_mad_1_mad_at")]
                return k_outer_info

        def _emit_insn_aub_bub():
            """
            do emit_insn for front tensor
            """
            if not self.binary_aub_process:
                return
            for tensor_name in binary_schedule.ub_tensor_list:
                tensor_a = self._grads_in_ub.get(tensor_name)
                tensor_b = self._fmap_in_ub.get(tensor_name)
                if tensor_name == "transpose_hw_c0":
                    sch[tensor_a].emit_insn(tensor_a.op.axis[-2], binary_schedule.emit_insn_dict.get(tensor_name))
                    sch[tensor_b].emit_insn(tensor_b.op.axis[-2], binary_schedule.emit_insn_dict.get(tensor_name))
                else:
                    sch[tensor_a].emit_insn(tensor_a.op.axis[0], binary_schedule.emit_insn_dict.get(tensor_name))
                    sch[tensor_b].emit_insn(tensor_b.op.axis[0], binary_schedule.emit_insn_dict.get(tensor_name))

        def _get_value(ele):
            res_ele = [ele.value if isinstance(ele, tvm.tir.IntImm) else ele][0]
            return res_ele

        def _get_load3d_para():
            # load_3d parameters
            if self.var_map and not \
                                DynamicConv2dBpFilterParams.flag_all_one_case:
                stride_height, stride_width, pad_up, pad_down, pad_left, \
                pad_right, kernel_height, kernel_width, dilation_height, \
                dilation_width = (
                    _get_value(self.tensor_map.get("fmap_fractal").op.attrs['stride'][0]),
                    _get_value(self.tensor_map.get("fmap_fractal").op.attrs['stride'][1]),
                    _get_value(self.tensor_map.get("fmap_fractal").op.attrs['pad'][0]),
                    _get_value(self.tensor_map.get("fmap_fractal").op.attrs['pad'][1]),
                    _get_value(self.tensor_map.get("fmap_fractal").op.attrs['pad'][2]),
                    _get_value(self.tensor_map.get("fmap_fractal").op.attrs['pad'][3]),
                    _get_value(self.tensor_map.get("fmap_fractal").op.attrs['kernel_size'][2]),
                    _get_value(self.tensor_map.get("fmap_fractal").op.attrs['kernel_size'][3]),
                    _get_value(self.tensor_map.get("fmap_fractal").op.attrs['dilation'][2]),
                    _get_value(self.tensor_map.get("fmap_fractal").op.attrs['dilation'][3])
                    )
            else:
                stride_height, stride_width, pad_up, pad_down, pad_left, \
                pad_right, kernel_height, kernel_width, dilation_height, \
                dilation_width = (
                    _get_value(self.tensor_map.get("fmap_matrix").op.attrs['stride'][0]),
                    _get_value(self.tensor_map.get("fmap_matrix").op.attrs['stride'][1]),
                    _get_value(self.tensor_map.get("fmap_matrix").op.attrs['pad'][0]),
                    _get_value(self.tensor_map.get("fmap_matrix").op.attrs['pad'][1]),
                    _get_value(self.tensor_map.get("fmap_matrix").op.attrs['pad'][2]),
                    _get_value(self.tensor_map.get("fmap_matrix").op.attrs['pad'][3]),
                    _get_value(self.tensor_map.get("fmap_matrix").op.attrs['kernel_size'][2]),
                    _get_value(self.tensor_map.get("fmap_matrix").op.attrs['kernel_size'][3]),
                    _get_value(self.tensor_map.get("fmap_matrix").op.attrs['dilation'][2]),
                    _get_value(self.tensor_map.get("fmap_matrix").op.attrs['dilation'][3])
                    )
            return stride_height, stride_width, pad_up, pad_down, pad_left, \
                   pad_right, kernel_height, kernel_width, dilation_height, \
                   dilation_width

        def _set_var_range():
            if self.binary_mode:
                binary_schedule.set_shape_var_range()
                binary_schedule.set_tiling_var_range()

        def _get_tiling():
            # get tiling from tilingcase
            tiling = dynamic_para.get('tiling')
            if self.var_map:
                # updata tiling in binary dynamic scene
                binary_schedule.config_cache_tiling(tiling)
                return tiling

            # Current design does not support multi load of Cin and Ho in BL1, which could lead to precision failure.
            # This limitation should be removed in future scheme.
            if self.flag_load3d_w_split_case == 1 and in_dtype == "float32":
                tiling["BL1_shape"][0] = tiling.get("BL0_matrix")[0] * self.c0_size
            elif (self.flag_load3d_w_split_case == 1 and
                tiling.get("BL1_shape")[1] * tiling.get("BL0_matrix")[1] > kernel_width * kernel_height and
                tiling.get("BL1_shape")[0] > tiling.get("BL0_matrix")[0] * self.c0_size):
                tiling["BL0_matrix"][1] = min(kernel_width * kernel_height, tiling.get("BL0_matrix")[1])
                tiling["CL0_matrix"][0] = min(kernel_width * kernel_height, tiling.get("CL0_matrix")[0])
                tiling["LOC_OUTPUT_matrix"][0] = min(kernel_width * kernel_height, tiling.get("LOC_OUTPUT_matrix")[0])
                tiling["BL1_shape"][1] = kernel_width * kernel_height // tiling.get("BL0_matrix")[1]

            return tiling

        def _handle_tbe_compile_para():
            tbe_compile_para = tiling.get("tbe_compile_para")
            sch.tbe_compile_para, tbe_sch_control_para = parse_tbe_compile_para(tbe_compile_para)
            preload_flag = tbe_sch_control_para.get("preload")
            reverse_load_flag = tbe_sch_control_para.get("reverse_load")
            if preload_flag and (tiling.get("manual_pingpong_buffer").get("CL0_pbuffer") == 2):
                sch[dw_cc].preload()
            # reverse is conflit with nbuffer
            a_overhead_opt_flag = bool(tiling.get("special_optimize_flag") % self.exponent_base)
            b_overhead_opt_flag = bool(tiling.get("special_optimize_flag") // self.exponent_base % self.exponent_base)
            self.reverse_params["reverse_load"] = reverse_load_flag and (
                (not a_overhead_opt_flag) and (not b_overhead_opt_flag))

        def _get_attach_flag():
            dynamic_l0a_attach = None
            dynamic_l0b_attach = None
            dynamic_al1_attach = None
            dynamic_bl1_attach = None
            attach_list = [dynamic_l0a_attach, dynamic_l0b_attach,
                           dynamic_al1_attach, dynamic_bl1_attach]
            if self.binary_mode:
                #  true or false
                dynamic_al1_attach = tiling.get("attach_at_flag").get("al1_attach_flag") in \
                    binary_schedule.k_full_load_list
                dynamic_bl1_attach = tiling.get("attach_at_flag").get("bl1_attach_flag") in \
                    binary_schedule.k_full_load_list
                dynamic_l0a_attach = dynamic_al1_attach and dynamic_bl1_attach and \
                    tiling.get("attach_at_flag").get("min_kl1_cmp_kl0") == 0
                dynamic_l0b_attach = dynamic_l0a_attach
                attach_list = [dynamic_l0a_attach, dynamic_l0b_attach,
                               dynamic_al1_attach, dynamic_bl1_attach]
                for item, val in enumerate(attach_list):
                    attach_list[item] = "dw_ddr" if val else "dw_cc"
            return attach_list

        def _ub_tensor_process(buffer_type, parent, attach_axis, bounds):
            ub_tensors = self._grads_in_ub if buffer_type == "aub" else self._fmap_in_ub
            for tensor_name in binary_schedule.ub_tensor_list:
                tensor = ub_tensors.get(tensor_name)
                sch[tensor].set_scope(tbe_platform_info.scope_ubuf)
                sch[tensor].compute_at(sch[parent], attach_axis)
                if self.binary_mode == cube_util.BinaryMode.NCHW:
                    if tensor_name == "transpose_hw_c0":
                        sch[tensor].compute_align(tensor.op.axis[2], CUBE_DIM) # 2 is the index of hw axis
                        sch[tensor].storage_align(tensor.op.axis[1], CUBE_MUL_SHAPE, 0) # 1 is the index of c1 axis
                        sch[tensor].buffer_tile(*bounds.get('nc1hwc0'))
                    else:
                        sch[tensor].compute_align(tensor.op.axis[2], CUBE_DIM) # 2 is the index of hw axis
                        sch[tensor].storage_align(tensor.op.axis[1], CUBE_DIM, 0) # 1 is the index of c axis
                        sch[tensor].buffer_tile(*bounds.get('nchw'))
                elif self.binary_mode == cube_util.BinaryMode.NHWC:
                    if tensor_name == "transpose_hw_c1":
                        sch[tensor].buffer_tile(*bounds.get('nc1hwc0'))

        def _split_binary_al1_process():
            """
            binary dynaimc 5HD wout=1 scene split hw axis
            """
            if  self.binary_mode == cube_util.BinaryMode.NC1HWC0 and tiling.get("load3d_special_flag") == 1:
                hw_outer, hw_inner = sch[self.tensor_map.get("grads_matrix")].split(
                    sch[self.tensor_map.get("grads_matrix")].op.axis[2], shape_expand_time)
                sch[self.tensor_map.get("grads_matrix")].reorder(hw_inner,
                    sch[self.tensor_map.get("grads_matrix")].op.axis[0],
                    sch[self.tensor_map.get("grads_matrix")].op.axis[1],
                    hw_outer, sch[self.tensor_map.get("grads_matrix")].op.axis[3])

        def _split_aub_process():
            """
            binary dynaimc scene has ub process
            """
            if not self.binary_aub_process:
                return
            c1_outer, c1_inner = sch[self.tensor_map.get("grads_matrix")].split(
                sch[self.tensor_map.get("grads_matrix")].op.axis[1], tiling.get('AUB_shape')[1])
            hw_outer, hw_inner = sch[self.tensor_map.get("grads_matrix")].split(
                sch[self.tensor_map.get("grads_matrix")].op.axis[2], tiling.get('AUB_shape')[0])
            if tiling.get("load3d_special_flag") == 1:
                hw_outer_outer, hw_outer_inner = sch[self.tensor_map.get("grads_matrix")].split(
                    hw_outer, shape_expand_time)
                hw_inner_outer, hw_inner_inner = sch[self.tensor_map.get("grads_matrix")].split(
                    hw_inner, shape_expand_time)
                sch[self.tensor_map.get("grads_matrix")].reorder(
                    c1_outer, hw_outer_outer, hw_outer_inner, hw_inner_inner,
                    sch[self.tensor_map.get("grads_matrix")].op.axis[0], c1_inner, hw_inner_outer)
                hw_single_core_factor = (binary_schedule.k_expr_single_core // shape_expand_time) * CUBE_DIM
            else:
                sch[self.tensor_map.get("grads_matrix")].reorder(
                    c1_outer, hw_outer, sch[self.tensor_map.get("grads_matrix")].op.axis[0], c1_inner, hw_inner)
                hw_single_core_factor = binary_schedule.k_expr_single_core  * CUBE_DIM
            sch[self.tensor_map.get("grads")].compute_inline()
            sch[self._grads_in_ub.get("reshape_c")].compute_inline()
            input_ub_vn = self._grads_in_ub.get("input_ub_vn")
            input_ub = self._grads_in_ub.get("input_ub_td")
            input_ub_pad = self._grads_in_ub.get("input_ub_pad")
            sch[input_ub_vn].reused_by(input_ub, input_ub_pad)

            k_bind_axis = self.sch[self.tensor_map.get("dw_ddr")].op.reduce_axis[0] // \
                ceil(batch_fmap, self.tiling_para_dict.get("batch_dim_factor"))
            # k_axis full load outer axis only block_idx
            hw_offset = k_bind_axis * hw_single_core_factor
            # split k, offset add al1.outer axis
            if dynamic_al1_attach != "dw_ddr":
                akl1_outer_offset = self.axis_dict.get("al1_at_axis") * tiling.get("AL1_shape")[0] // shape_expand_time
                hw_offset = hw_offset + akl1_outer_offset
                # kbl1 > kal1, offset add bl1.outer
                if reduce_split_mode:
                    bkl1_outer_offset = self.axis_dict.get("bl1_at_axis") * tiling.get("BL1_shape")[0]
                    hw_offset = hw_offset + bkl1_outer_offset
            attach_axis = hw_outer if tiling.get("load3d_special_flag") != 1 else hw_outer_outer
            hw_offset = hw_offset + attach_axis * tiling.get('AUB_shape')[0]

            bounds = {
                "nchw": ((None, None), (None, None), (hw_offset, tiling.get("AUB_shape")[0])),
                "nc1hwc0": ((None, None), (None, None),
                            (hw_offset, tiling.get("AUB_shape")[0]), (None, None)),
            }
            _ub_tensor_process("aub", self.tensor_map.get("grads_matrix"), attach_axis, bounds)

        def _split_bub_process():
            """
            binary dynaimc scene has ub process
            """
            if not self.binary_bub_process:
                return
            c_idx = 1 if DynamicConv2dBpFilterParams.flag_all_one_case else 2
            h_factor = tiling.get('BUB_shape')[0]
            w_factor = tvm.select(tvm.all(tiling.get("BUB_shape")[2] == width_fmap),
                                  stride_width * (width_grads - 1) + kernel_width,
                                  tiling.get("BUB_shape")[2])
            reorder_lis = [sch[self.tensor_map.get("fmap_matrix")].op.axis[0],
                           sch[self.tensor_map.get("fmap_matrix")].op.axis[1]]
            c1_outer, c1_inner = sch[self.tensor_map.get("fmap_matrix")].split(
                sch[self.tensor_map.get("fmap_matrix")].op.axis[c_idx], tiling.get('BUB_shape')[1])
            h_outer = 0
            w_outer = 0
            if DynamicConv2dBpFilterParams.flag_all_one_case:
                # flag_all_one_case, fmap_matrix shape is [batch, fmap_c1, fmap_h*fmap_w, fmap_c0]
                c_idx = 1
                k_idx = 2
                hw_factor = tvm.min(h_factor * w_factor,
                                    binary_schedule.cache_tiling["kbl1_16"] * CUBE_DIM)
                reorder_lis = [
                    sch[self.tensor_map.get("fmap_matrix")].op.axis[0], ]
                h_outer, h_inner = sch[self.tensor_map.get("fmap_matrix")].split(
                    sch[self.tensor_map.get("fmap_matrix")].op.axis[k_idx], hw_factor)
                sch[self.tensor_map.get("fmap_matrix")].reorder(
                    c1_outer, h_outer, *reorder_lis, c1_inner, h_inner)
            else:
                # Base scene, fmap_matrix shape is [real_g, batch_size, fmap_c1_g, fmap_h, fmap_w, fmap_c0]
                h_outer, h_inner = sch[self.tensor_map.get("fmap_matrix")].split(
                    sch[self.tensor_map.get("fmap_matrix")].op.axis[3], h_factor)
                w_outer, w_inner = sch[self.tensor_map.get("fmap_matrix")].split(
                    sch[self.tensor_map.get("fmap_matrix")].op.axis[4], w_factor)
                sch[self.tensor_map.get("fmap_matrix")].reorder(
                    c1_outer, h_outer, w_outer, *reorder_lis, c1_inner, h_inner, w_inner)

            sch[self.tensor_map.get("fmap")].compute_inline()
            sch[self._fmap_in_ub.get("reshape_c")].compute_inline()
            input_ub_vn = self._fmap_in_ub.get("input_ub_vn")
            input_ub = self._fmap_in_ub.get("input_ub_td")
            input_ub_pad = self._fmap_in_ub.get("input_ub_pad")
            sch[input_ub_vn].reused_by(input_ub, input_ub_pad)
            hw_offset, once_dma_data = _tensor_b_buffer_tile_in_binary(h_outer, w_outer)
            bounds = {
                "nchw": ((None, None), (None, None), (hw_offset, once_dma_data)),
                "nc1hwc0": ((None, None), (None, None), (hw_offset, once_dma_data), (None, None))
            }
            if DynamicConv2dBpFilterParams.flag_all_one_case:
                _ub_tensor_process("bub", self.tensor_map.get("fmap_matrix"), h_outer, bounds)
            else:
                _ub_tensor_process("bub", self.tensor_map.get("fmap_matrix"), w_outer, bounds)

        def _tensor_b_buffer_tile_in_binary(h_outer, w_outer=0):
            """
            do buffer tile for BL1 and Bub
            """
            hw_single_core_factor = binary_schedule.k_expr_single_core * self.c0_size
            ho_bl1 = binary_schedule.cache_tiling.get("ho_bL1")
            # get ho_offset about BL1 according split of L1
            hw_offset = self.sch[self.tensor_map.get("dw_ddr")].op.reduce_axis[0] // \
                ceil(batch_fmap, self.tiling_para_dict.get("batch_dim_factor")) * hw_single_core_factor
            # split k, offset add bl1.outer axis and al1.outer in different scenes
            if dynamic_bl1_attach != "dw_ddr":
                bkl1_outer_offset = self.axis_dict.get("bl1_at_axis") * tiling.get("BL1_shape")[0]
                hw_offset = hw_offset + bkl1_outer_offset
                if not reduce_split_mode:
                    akl1_outer_offset = self.axis_dict.get("al1_at_axis") * tiling.get("AL1_shape")[0]
                    hw_offset = hw_offset + akl1_outer_offset
            ho_offset = hw_offset // width_grads
            # get hi_offset about BL1 according ho_offset and compute expression
            hi_offset = ho_offset * stride_height - pad_up
            hi_extend = (ho_bl1 - 1) * stride_height + kernel_height
            if DynamicConv2dBpFilterParams.flag_all_one_case:
                hw_extent = binary_schedule.cache_tiling["kbl1_16"] * self.c0_size
                # fmap_matrix shape is [batch, fmap_c1, fmap_h*fmap_w, fmap_c0]
                sch[self.tensor_map.get("fmap_matrix")].buffer_tile((None, None), (None, None),
                                                                    (hw_offset, hw_extent), (None, None))
                bub_k_factor = tvm.min(tiling.get('BUB_shape')[0] * tiling.get("BUB_shape")[2],
                                       binary_schedule.cache_tiling["kbl1_16"] * self.c0_size)
                return _bub_buffer_tile_param(h_outer, hw_offset, bub_k_factor)
            elif flag_conv1d_case:
                wo_offset = hw_offset
                wi_offset = wo_offset * stride_width
                bub_k_factor = tiling.get("BUB_shape")[0] * tiling.get("BUB_shape")[2]
                hiwi_offset = wi_offset
                return _bub_buffer_tile_param(h_outer, hiwi_offset, bub_k_factor, w_outer)
            else:
                # fmap_l1 shape is [real_g, batch_size, fmap_c1_g, fmap_h, fmap_w, fmap_c0]
                sch[self.tensor_map.get("fmap_l1")].buffer_tile((None, None), (None, None), (None, None),
                                         (hi_offset, hi_extend), (None, None), (None, None))
                bub_k_factor = tiling.get("BUB_shape")[0] * tiling.get("BUB_shape")[2]
                hiwi_offset = hi_offset * width_fmap
                return _bub_buffer_tile_param(h_outer, hiwi_offset, bub_k_factor, w_outer)

        def _bub_buffer_tile_param(h_outer, hiwi_offset, bub_k_factor, w_outer=0):
            """
            get bub buffer_tile param
            """
            # The offset of the main block is front_hw_offset
            # The offset of the tail block is height_fmap * width_fmap - bub_extent
            bub_extent = (bub_k_factor + CUBE_DIM - 1) // CUBE_DIM * CUBE_DIM
            front_hw_offset = 0
            if DynamicConv2dBpFilterParams.flag_all_one_case:
                front_hw_offset = hiwi_offset + h_outer * bub_k_factor
            else:
                front_hw_offset = hiwi_offset + h_outer * tiling.get("BUB_shape")[0] * width_fmap + \
                    w_outer * tiling.get("BUB_shape")[2]
            hw_offset = tvm.select(tvm.all(tiling.get("BUB_shape")[2] == width_fmap),
                                   tvm.min(front_hw_offset + bub_extent, height_fmap * width_fmap) - bub_extent,
                                   tvm.min(front_hw_offset, height_fmap * width_fmap) - pad_left)
            hw_offset = tvm.max(hw_offset, tvm.const(0))
            return hw_offset, bub_extent

        def _set_bound_ub():
            """
            set bound for ub tensor
            """
            if not self.binary_aub_process:
                return
            aub_bound = tiling.get("AUB_shape")[0] * tiling.get("AUB_shape")[1] * CUBE_DIM
            bub_bound = (tiling.get("BUB_shape")[0] * tiling.get("BUB_shape")[2] + CUBE_DIM - 1) \
                // CUBE_DIM * CUBE_DIM * tiling.get("BUB_shape")[1] * CUBE_DIM
            for tensor_name in binary_schedule.ub_tensor_list:
                tensor_a = self._grads_in_ub.get(tensor_name)
                tensor_b = self._fmap_in_ub.get(tensor_name)
                sch[tensor_a].set_buffer_size(aub_bound)
                sch[tensor_b].set_buffer_size(bub_bound)
            if self._grads_in_ub.get("transpose_hw_c0") is not None:
                sch[self._grads_in_ub.get("transpose_hw_c0")].mem_unique()
                sch[self._fmap_in_ub.get("transpose_hw_c0")].mem_unique()
            elif self._grads_in_ub.get("transpose_hw_c1") is not None:
                sch[self._grads_in_ub.get("transpose_hw_c1")].mem_unique()
                sch[self._fmap_in_ub.get("transpose_hw_c1")].mem_unique()

        # for dynamic
        self.var_map = DynamicConv2dBpFilterParams.var_map

        # ####################### get computing graph #######################
        ## (1) input dtype: float16
        # orig:
        #   dw_ddr -> (dw_res_trans)
        # atomic_add:
        #   dw_ddr.rf -> (dw_ub) -> dw_ddr -> (dw_res_trans)
        # inline 'dw_ddr' (optional):
        #   dw_ddr.rf -> (dw_ub) -> dw_res_trans

        ## (2) input dtype: float32
        # orig:
        #   dw_ddr -> dw_ddr_c_split -> (dw_res_trans)
        # atomic_add:
        #   dw_ddr.rf -> (dw_ub) -> dw_ddr -> dw_ddr_c_split -> (dw_res_trans)
        # inline 'dw_ddr':
        #   dw_ddr.rf -> (dw_ub) -> dw_ddr_c_split -> (dw_res_trans)
        # inline 'dw_ddr_c_split' (optional):
        #   dw_ddr.rf -> (dw_ub) -> dw_res_trans

        ## (3) with fixpipe, input dtype: float32
        # orig:
        #   dw_ddr -> dw_ddr_c_split
        # fixpipe fusion:
        #   dw_ddr -> fixpipe_op -> fixpipe_reform
        # atomic_add:
        #   dw_ddr.rf -> dw_ddr -> fixpipe_op -> fixpipe_reform
        # inline 'dw_ddr':
        #   dw_ddr.rf -> fixpipe_op -> fixpipe_reform
        # inline 'fixpipe_op':
        #   dw_ddr.rf -> fixpipe_reform

        sch = sch_list[0]
        # use ScheduleAgentReverse for split dw_cc and dw_ddr's axes, record split factor while split axis
        sch_agent = conv2dBpFilterUtil.ScheduleAgentReverse(sch)
        # for binary dynamic
        binary_schedule = Conv2DBpFilterBinaryDynamic(sch, self.binary_mode, self.flag_load3d_w_split_case,
                                                      DynamicConv2dBpFilterParams.flag_all_one_case)
        self.sch = sch
        self.sch_agent = sch_agent
        self.binary_schedule = binary_schedule

        if res.op.tag == "FZ_trans_NHWC":
            self.dw_trans_flag = True
            self.tensor_map["dw_res_trans"] = res
            self.tensor_map["dw_ddr"] = res.op.input_tensors[0]
        else:
            self.tensor_map["dw_ddr"] = res

        # support channel split
        # dw_cc -> dw_ddr
        if self.tensor_map.get("dw_ddr").op.tag == "conv2d_backprop_filter_c_split":
            self.c_split_flag = True
            dw_cc = self.tensor_map.get("dw_ddr").op.input_tensors[0]
        elif self.tensor_map.get("dw_ddr").op.tag == "fixpipe_reform":
            self.dw_fixpipe_flag = True
            fixpipe_tensor = self.tensor_map.get("dw_ddr").op.input_tensors[0]
            dw_cc = fixpipe_tensor.op.input_tensors[0]

            vector_params = fixpipe_tensor.op.attrs["vector_params"]
            vector_tensors = fixpipe_tensor.op.attrs["vector_tensors"]
            fixpipe_fb_dict = {}
            fixpipe_l1_list = []

            for idx, params_mem in enumerate(vector_params):
                fixpipe_input = vector_tensors[idx]
                fixpipe_input_l1 = sch.cache_read(fixpipe_input, tbe_platform_info.scope_cbuf, [fixpipe_tensor])
                fixpipe_scope_name = conv2dBpFilterUtil.FIXPIPE_SCOPE_MAP.get(params_mem.value)
                if fixpipe_scope_name:
                    fixpipe_input_fb = sch.cache_read(fixpipe_input_l1, fixpipe_scope_name, [fixpipe_tensor])
                    fixpipe_l1_list.append(fixpipe_input_l1)
                    fixpipe_fb_dict[fixpipe_scope_name] = fixpipe_input_fb
                else:
                    self.tensor_map["fixpipe_l1_eltwise"] = fixpipe_input_l1
            self.tensor_map["fixpipe_tensor"] = fixpipe_tensor
            self.tensor_map["fixpipe_reform"] = self.tensor_map.get("dw_ddr")
            self.tensor_map["fixpipe_fb"] = fixpipe_fb_dict
            self.tensor_map["fixpipe_l1"] = fixpipe_l1_list
        else:
            dw_cc = self.tensor_map.get("dw_ddr")

        self.tensor_map["grads_fractal"] = dw_cc.op.input_tensors[0]
        self.tensor_map["grads_matrix"] = self.tensor_map.get("grads_fractal").op.input_tensors[0]
        self.tensor_map["grads"] = self.tensor_map.get("grads_matrix").op.input_tensors[0]
        self.tensor_map["fmap_fractal"] = dw_cc.op.input_tensors[1]
        if self.tensor_map.get("fmap_fractal").op.tag == "fmap_2_fractal_dma":
            if self.var_map:
                self.tensor_map["fmap_matrix"] = self.tensor_map.get("fmap_fractal").op.input_tensors[0]
            else:
                fmap_fractal_before = self.tensor_map.get("fmap_fractal").op.input_tensors[0]
                self.tensor_map["fmap_matrix"] = fmap_fractal_before.op.input_tensors[0]
            self.l0b_dma_flag = True
        else:
            self.tensor_map["fmap_matrix"] = self.tensor_map.get("fmap_fractal").op.input_tensors[0]
        self.tensor_map["fmap"] = self.tensor_map.get("fmap_matrix").op.input_tensors[0]
        if self.tensor_map.get("fmap").op.tag == "fmap_ub_for_dma":
            self.tensor_map["fmap_ub"] = self.tensor_map.get("fmap_matrix").op.input_tensors[0]
            self.tensor_map["fmap"] = self.tensor_map.get("fmap_ub").op.input_tensors[0]
        self.tensor_map["fmap_l1"] = _get_c04_fmap_l1(self.tensor_map.get("fmap_fractal"), self._c04_flag)
        if not self.var_map and self._strideh_read_flag:
            self.tensor_map["fmap_l1_name"] = self.tensor_map.get("fmap_matrix").op.input_tensors[0]
            self.tensor_map["fmap"] = self.tensor_map.get("fmap_l1_name").op.input_tensors[0]

        _get_previous_ub_list(self.tensor_map.get("fmap"), self._fmap_in_ub)
        _get_previous_ub_list(self.tensor_map.get("grads"), self._grads_in_ub)

        if self.tensor_map.get("fmap").op.tag == "NHWC_trans_5HD":
            self.fmap_trans_flag = True
        if self.tensor_map.get("grads").op.tag == "NHWC_trans_5HD":
            self.grads_trans_flag = True
        self.binary_aub_process = not (not self.binary_mode or self.binary_mode == cube_util.BinaryMode.NC1HWC0
                                       or self.grads_trans_flag)
        self.binary_bub_process = not (not self.binary_mode or self.binary_mode == cube_util.BinaryMode.NC1HWC0
                                       or self.fmap_trans_flag)

        self._show_flags()
        self.group_dict = self.tensor_map.get("fmap_matrix").op.attrs['group_dict']

        # fmap_dtype is same as outbackprop, use fmap_dtype to get C0 size
        in_dtype = self.tensor_map.get("fmap").dtype.lower()
        if in_dtype == "float32":
            # mac[1] meas k axis
            self.c0_size = tbe_platform.CUBE_MKN.get(in_dtype).get("mac")[1]

        # ########################extract parameters##########################
        # type of group_dict is tvm.container.StrMap not dict
        cin1_g = _get_value(self.group_dict["cin1_g"])
        cout_g = _get_value(self.group_dict["cout_g"])
        real_g = _get_value(self.group_dict["real_g"])

        batch_grads, c1_grads, height_grads, width_grads, c0_grads = \
            cube_util.shape_to_list(self.tensor_map.get("grads").shape)
        grads_shape = [batch_grads, cout_g // c0_grads, height_grads, width_grads, c0_grads]
        batch_fmap, c1_fmap, height_fmap, width_fmap, c0_fmap = \
            cube_util.shape_to_list(self.tensor_map.get("fmap").shape)

        if not self.flag_load3d_w_split_case:
            _, grads_matrix_c1, grads_matrix_howo, grads_matrix_c0 = \
                cube_util.shape_to_list(self.tensor_map.get("grads_matrix").shape)

        if self.flag_load3d_w_split_case == 1:
            _, _, _, wo_mad_1, _, _, _ = cube_util.shape_to_list(self.tensor_map.get("fmap_fractal").shape)
        else:
            _, _, hw_pad_1, _, _, _ = cube_util.shape_to_list(self.tensor_map.get("fmap_fractal").shape)

        (stride_height, stride_width, pad_up, pad_down, pad_left, pad_right, kernel_height, kernel_width,
         dilation_height, dilation_width) = _get_load3d_para()

        featuremap_channel = c1_fmap * c0_fmap
        featuremap_height = height_fmap
        featuremap_width = width_fmap
        kw_dilation = (kernel_width - 1) * dilation_width + 1
        weight_shape = [cout_g, cin1_g, kernel_height, kernel_width, c0_fmap]

        _set_var_range()

        def _flag_all_one():
            # special supporting for a unique case, there are 2 conditions:
            # (1) height & weight of x/output_backprop/filter are all 1
            # (2) strides is [1,1]
            flag_all_one_case = DynamicConv2dBpFilterParams.flag_all_one_case
            flag_conv1d_case = DynamicConv2dBpFilterParams.conv1d_flag
            flag_w_one_case = False

            if height_grads != 1 and width_grads == 1:
                flag_w_one_case = True
            if self.binary_mode and not flag_all_one_case:
                flag_conv1d_case = bool(dynamic_para.get("tiling").get('conv1d_flag', False))
            return flag_all_one_case, flag_conv1d_case, flag_w_one_case

        flag_all_one_case, flag_conv1d_case, flag_w_one_case = _flag_all_one()
        tiling_dtype = get_inout_dtype(self.tensor_map.get(
            "grads"), self.tensor_map.get("fmap"), dw_cc, "Conv2DBackpropFilter")
        tiling = _get_tiling()
        kernel_name = self.tensor_map.get("dw_ddr").op.attrs["kernel_name"]
        log.debug("[{}] kernel_name = {}, tiling_key = {}, tiling = {}".format(
            CONV2D_BACKPROP_FILTER_OP_NAME, kernel_name, dynamic_para.get('tiling_key'), str(tiling)))
        dynamic_l0a_attach, dynamic_l0b_attach, dynamic_al1_attach, dynamic_bl1_attach = _get_attach_flag()

        # for dynamic_mode w_one_case
        if self.var_map:
            flag_w_one_case = tiling.get("load3d_special_flag") == 1
            if flag_w_one_case:
                width_grads *= 2
                grads_shape[3] = width_grads

        _tiling_shape_check()
        _tiling_buffer_check()

        batch_num = batch_grads

        def _get_block_dim():
            if self.binary_mode:
                block_dim_k = binary_schedule.cache_tiling.get("k_dim")
            elif tiling.get("AUB_shape"):
                block_dim_k = tiling.get("AUB_shape")[0]
            else:
                block_dim_k = 1
            block_dim_batch = tiling.get("block_dim")[0]
            block_dim_cout = tiling.get("block_dim")[2]
            block_dim_cin = tiling.get("block_dim")[1]
            block_dim_group = tiling.get("block_dim")[3]
            self.tiling_para_dict["n_single_core"] = ceil(cin1_g, block_dim_cin) * kernel_height * kernel_width
            if self.c_split_flag:
                real_cin_kh_kw = cube_util.shape_to_list(self.tensor_map.get("dw_ddr").shape)[1]
                real_cin = real_cin_kh_kw // kernel_height // kernel_width
                singlecore_cin = ceil(real_cin, block_dim_cin)
                singlecore_cin = singlecore_cin if block_dim_cin == 1 else align(singlecore_cin, 2)
                self.tiling_para_dict["n_single_core"] = singlecore_cin * kernel_height * kernel_width
            if self._c04_flag:
                row_major_c0 = conv2dBpFilterUtil.C04_SIZE
                block_size = conv2dBpFilterUtil.BLOCK_SIZE
                self.tiling_para_dict["n_single_core"] = \
                    (self.tiling_para_dict.get("n_single_core") * row_major_c0 + block_size - 1) // block_size
            return (block_dim_k, block_dim_batch, block_dim_cout, block_dim_cin, block_dim_group)

        block_dim_k, block_dim_batch, block_dim_cout, block_dim_cin, block_dim_group = _get_block_dim()
        self.tiling_para_dict["block_dim"] = [block_dim_k, block_dim_batch,
                                              block_dim_cout, block_dim_cin, block_dim_group]

        mul_align_length = CUBE_MUL_SHAPE
        k_align_length = CUBE_DIM
        if in_dtype == "float32":
            # in float32 case, k axis length will be reduce by half each repeatition
            mul_align_length = CUBE_MUL_SHAPE // 2
            k_align_length = CUBE_DIM // 2

        if self.grads_trans_flag:
            sch[self.tensor_map.get("grads")].set_scope(tbe_platform_info.scope_cbuf)
        sch[self.tensor_map.get("grads_matrix")].set_scope(tbe_platform_info.scope_cbuf)
        sch[self.tensor_map.get("grads_fractal")].set_scope(tbe_platform_info.scope_ca)

        if self.flag_load3d_w_split_case == 1:
            # align axis is 2 in w-split situation
            # The BL1 shape in scenario flag_load3d_w_split_case is (n, co, ho, wo, c0)
            sch[self.tensor_map.get("grads_matrix")].storage_align(
                sch[self.tensor_map.get("grads_matrix")].op.axis[2], mul_align_length, 0)
            sch[self.tensor_map.get("grads_fractal")].buffer_align(
                (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, CUBE_DIM), (1, k_align_length))
        else:
            if in_dtype != "float32":
                # load3d(fp32) no need to be aligned
                sch[self.tensor_map.get("grads_matrix")].storage_align(
                    sch[self.tensor_map.get("grads_matrix")].op.axis[1], CUBE_MUL_SHAPE, 0)
            sch[self.tensor_map.get("grads_fractal")].buffer_align(
                (1, 1), (1, 1), (1, 1), (1, 1), (1, CUBE_DIM), (1, k_align_length))

        def _load3d_fmap_l1_process():
            # shape info:
            # fmap_shape_original_matrix is (batch_size,
            #                               grads_height*grads_width,
            #                               fmap_channel_1,
            #                               kernel_height,
            #                               kernel_width,
            #                               C0_fmap)
            if not self.var_map:
                if self._strideh_read_flag:
                    if self.fmap_trans_flag:
                        sch[self.tensor_map.get("fmap")].compute_inline()
                    self.tensor_map["fmap_l1"] = self.tensor_map.get("fmap_l1_name")
                    sch[self.tensor_map.get("fmap_l1")].set_scope(tbe_platform_info.scope_cbuf)
                elif self.fmap_trans_flag:
                    self.tensor_map["fmap_l1"] = self.tensor_map.get("fmap")
                    sch[self.tensor_map.get("fmap")].set_scope(tbe_platform_info.scope_cbuf)
                elif self._c04_flag:
                    sch[self.tensor_map.get("fmap_l1")].set_scope(tbe_platform_info.scope_cbuf)
                elif self.tensor_map.get("fmap_ub") is not None:
                    sch[self.tensor_map.get("fmap_ub")].set_scope(tbe_platform_info.scope_ubuf)
                    self.tensor_map["fmap_l1"] = sch.cache_read(
                        self.tensor_map.get("fmap_ub"), tbe_platform_info.scope_cbuf,
                        [self.tensor_map.get("fmap_matrix")])
                else:
                    self.tensor_map["fmap_l1"] = sch.cache_read(
                        self.tensor_map.get("fmap"), tbe_platform_info.scope_cbuf,
                        [self.tensor_map.get("fmap_matrix")])

                c0_align = conv2dBpFilterUtil.C04_SIZE if self._c04_flag else CUBE_DIM
                if flag_conv1d_case:
                    sch[self.tensor_map.get("fmap_matrix")].buffer_align(
                        (1, 1), (1, 1), (1, 1), (1, 1),
                        (kernel_width, kernel_width), (1, c0_align))
                elif self.flag_load3d_w_split_case == 1:
                    sch[self.tensor_map.get("fmap_matrix")].buffer_align(
                        (1, 1), (1, 1), (1, width_grads), (1, 1),
                        (kernel_height, kernel_height), (kernel_width, kernel_width), (1, c0_align))
                else:
                    sch[self.tensor_map.get("fmap_matrix")].buffer_align(
                        (1, 1), (width_grads, width_grads), (1, 1),
                        (kernel_height, kernel_height), (kernel_width, kernel_width), (1, c0_align))
            else:
                if self.tensor_map.get("fmap_ub") is not None:
                    sch[self.tensor_map.get("fmap_ub")].set_scope(tbe_platform_info.scope_ubuf)
                self.tensor_map["fmap_l1"] = self.tensor_map.get("fmap_matrix")

        if not flag_all_one_case:
            _load3d_fmap_l1_process()
        else:
            sch[self.tensor_map.get("fmap_matrix")].storage_align(
                sch[self.tensor_map.get("fmap_matrix")].op.axis[1], mul_align_length, 0)

        sch[self.tensor_map.get("fmap_matrix")].set_scope(tbe_platform_info.scope_cbuf)

        if self.l0b_dma_flag and not self.var_map:
            sch[fmap_fractal_before].set_scope(tbe_platform_info.scope_cbuf)
            sch[fmap_fractal_before].buffer_align((1, 1), (1, 1), (1, 1), (1, 1),
                                                  (1, CUBE_DIM), (1, k_align_length))

        sch[self.tensor_map.get("fmap_fractal")].set_scope(tbe_platform_info.scope_cb)
        if self.flag_load3d_w_split_case == 1:
            sch[self.tensor_map.get("fmap_fractal")].buffer_align(
                (1, 1), (1, 1), (1, 1), (1, 1), (1, 1),
                (1, CUBE_DIM), (1, k_align_length))
        else:
            sch[self.tensor_map.get("fmap_fractal")].buffer_align(
                (1, 1), (1, 1), (1, 1), (1, 1),
                (1, CUBE_DIM), (1, k_align_length))

        full_k_in_l0a, full_k_in_l0b = _full_k_check()

        _compute_tiling_parts()
        dw_tiling_factor = self.tiling_para_dict.get("dw_tiling_factor")
        dw_tiling_nparts = self.tiling_para_dict.get("dw_tiling_nparts")
        dw_ub_tiling_factor = self.tiling_para_dict.get("dw_ub_tiling_factor")

        grads_l1_tiling_nparts = self.tiling_para_dict.get("grads_l1_tiling_nparts")
        fmap_l1_tiling_nparts = self.tiling_para_dict.get("fmap_l1_tiling_nparts")

        l1_2_l0_tiling_nparts = self.tiling_para_dict.get("l1_2_l0_tiling_nparts")
        dw_k = self.tiling_para_dict.get("dw_k")
        reduce_split_mode = _reduce_split_mode()
        if tiling.get("AL1_shape") and len(tiling.get("AL1_shape")) > 2:
            # batch_al1 can be greater than 1
            self.batch_al1 = tiling.get("AL1_shape")[2]
        grads_l1_tiling_factor_k, fmap_l1_tiling_factor_k = _compute_tiling_factors()

        dw_cc, dw_ub, dw_rfactor = _atomic_add(sch, dw_cc)
        if self.dw_fixpipe_flag:
            # dw_ddr.rf -> dw_ddr -> fixpipe_tensor -> fixpipe_reform
            # after inline: dw_ddr.rf -> fixpipe_reform
            fixpipe_tensor = self.tensor_map.get("fixpipe_tensor")
            self.tensor_map["dw_ddr"] = fixpipe_tensor.op.input_tensors[0]
            sch[self.tensor_map.get("dw_ddr")].compute_inline(instant=True)
            sch[fixpipe_tensor].compute_inline(instant=True)

            self.tensor_map["dw_ddr"] = self.tensor_map.get("fixpipe_reform")
            self.dw_trans_flag = self.tensor_map.get("dw_ddr").op.name == "fixpipe_nz2nd"
            self.c_split_flag = self.tensor_map.get("dw_ddr").op.name == "fixpipe_channel_split"
        elif self.c_split_flag:
            # dw_ddr.rf -> dw_ddr -> dw_c_split
            # after inline: dw_ddr.rf -> dw_c_split
            sch[dw_rfactor].compute_inline(instant=True)
        else:
            # dw_ddr.rf -> dw_ddr
            self.tensor_map["dw_ddr"] = dw_rfactor
        # #######################tiling parameters analyze####################
        batch_num_sc = ceil(batch_num, block_dim_batch)
        if self.batch_al1 > 1:
            # batch_ddr is performed on the ddr by atomic add operation, batch for l0c can be regarded as batch_al1
            batch_num_sc = self.batch_al1

        load_order_change_flag = _load_order_change_flag()

        def _get_n_factor():
            # for N axis, if Hk and Wk needs split, do explict split
            if not flag_all_one_case:
                if tiling.get("BL1_shape"):
                    # n1 in L1
                    nc_cc = self.tiling_para_dict.get("dw_tiling_factor")[0] * tiling.get("BL1_shape")[1]
                else:
                    # BL1 is full load
                    nc_cc = self.tiling_para_dict.get("n_single_core")
                factor_kw = ceil(kernel_width, nc_cc)
                factor_kh = ceil(kernel_width*kernel_height, nc_cc) // factor_kw
            else:
                factor_kw = 1
                factor_kh = 1
            return factor_kw, factor_kh

        factor_kw, factor_kh = _get_n_factor()

        # #############################split axis N##########################
        self._dw_ddr_split(kernel_width, kernel_height, factor_kw, factor_kh)

        if self.batch_al1 > 1:
            # ractor will cause condition loss. in batch_al1 scence, there is a part of batch axis do
            # atomic add in ddr, need to add the condition manually
            batch_axis_range = block_dim_batch * self.tiling_para_dict.get("batch_dim_factor") // self.batch_al1
            sch[self.tensor_map.get("dw_ddr")].set_store_predicate(
                sch[self.tensor_map.get("dw_ddr")].op.reduce_axis[0]
                    % batch_axis_range * self.batch_al1 < batch_fmap, True
            )

        def _allocate_at_split():
            run_once_ndim = []
            c_fmap_mad_at = self.axis_dict.get("c_fmap_mad_at")
            c_fmap_mad_at_list = [c_fmap_mad_at, ]
            factor_c = ceil(ceil(cin1_g, block_dim_cin) * kernel_height * kernel_width, dw_tiling_nparts[0])
            b_overhead_opt_flag = bool(tiling.get("special_optimize_flag") // self.exponent_base % self.exponent_base)
            if b_overhead_opt_flag:
                nbuffer_size = kernel_height * kernel_width // factor_c
                if kernel_height * kernel_width % factor_c == 0 and tiling.get("BL1_shape")[1] % nbuffer_size == 0:
                    c_fmap_run_once, c_fmap_mad_at \
                        = sch_agent[self.tensor_map.get("dw_ddr")].split(c_fmap_mad_at, nbuffer_size)
                    run_once_ndim = [c_fmap_mad_at, ]
                    c_fmap_mad_at_list = [c_fmap_run_once, c_fmap_mad_at]
            return run_once_ndim, c_fmap_mad_at_list, c_fmap_mad_at
        run_once_ndim, c_fmap_mad_at_list, c_fmap_mad_at = _allocate_at_split()
        self.axis_dict["c_fmap_mad_at"] = c_fmap_mad_at

        def _cub_and_cc_attach():
            # optimization by move small loops to outer
            if not self.binary_mode:
                l0b_attach_mode = (batch_num_sc == 1 and full_k_in_l0b == 1)
                m_block = tiling.get("CL0_matrix")[1]
                n_block = tiling.get("CL0_matrix")[0]
                k_block = tiling.get("AL0_matrix")[1]
                byte_size = DTYPE_BYTE_MAPPING.get(in_dtype, 2)
                l1_2_l0a_rate = int(tbe_platform_info.get_soc_spec("l1_to_l0_a_rate"))
                l1_2_l0b_rate = int(tbe_platform_info.get_soc_spec("l1_to_l0_b_rate"))
                if l1_2_l0a_rate == 0 or l1_2_l0b_rate == 0:
                    dict_args = {'errCode': "E60108", 'reason': "Get l1 to l0 rate error"}
                    error_manager_util.raise_runtime_error(dict_args)
                extra_l0a = (l1_2_l0_tiling_nparts[0] - 1) * (LOAD2D_CONSUME + m_block * k_block *
                                                              CUBE_MUL_SHAPE * byte_size // l1_2_l0a_rate)
                extra_l0b = (l1_2_l0_tiling_nparts[1] - 1) * (LOAD3D_CONSUME + n_block * k_block *
                                                              CUBE_MUL_SHAPE * byte_size // l1_2_l0b_rate)
                reorder_flag = l0b_attach_mode and (extra_l0a > extra_l0b)
                reorder_l1_mn = fmap_l1_tiling_nparts[1] > grads_l1_tiling_nparts[1]
            else:
                reorder_flag = False
                reorder_l1_mn = False
            # during L1 to L0, if M loop is smaller, then move to outer
            if reorder_flag:
                sch[self.tensor_map.get("dw_ddr")].reorder(self.axis_dict.get("c_grads_mad_at"), *c_fmap_mad_at_list)
            # during sc to L1, if M loop is smaller, then move to outer
            if reorder_l1_mn:
                sch[self.tensor_map.get("dw_ddr")].reorder(
                    self.axis_dict.get("c_grads_l1_at"), self.axis_dict.get("c_fmap_l1_c1"),
                    self.axis_dict.get("c_fmap_l1_kh"), self.axis_dict.get("c_fmap_l1_at"))

            if not self.support_l0c2out:
                # dw_ub attach
                # dw_ub split
                c_fmap_2_ub_at, c_fmap_2_ub_insn \
                    = sch_agent[self.tensor_map.get("dw_ddr")].split(
                        self.axis_dict.get("c_fmap_mad_insn"), dw_ub_tiling_factor[0],
                        split_params=self.binary_schedule.split_params)
                # dw_ub attach
                sch[dw_ub].compute_at(sch[self.tensor_map.get("dw_ddr")], c_fmap_2_ub_at)
            else:
                c_fmap_2_ub_insn = self.axis_dict.get("c_fmap_mad_insn")

            # dw attach
            if reorder_flag:
                sch[dw_cc].compute_at(sch[self.tensor_map.get("dw_ddr")], self.axis_dict.get("c_fmap_mad_at"))
                self.l0c_attach_axis = self.axis_dict.get("c_fmap_mad_at")
            else:
                sch[dw_cc].compute_at(sch[self.tensor_map.get("dw_ddr")], self.axis_dict.get("c_grads_mad_at"))
                self.l0c_attach_axis = self.axis_dict.get("c_grads_mad_at")
            return c_fmap_2_ub_insn, reorder_flag, reorder_l1_mn
        c_fmap_2_ub_insn, reorder_flag, reorder_l1_mn = _cub_and_cc_attach()

        self._dw_cc_split(dw_cc, reduce_split_mode, load_order_change_flag)

        # #############################multi core#############################
        blocks = (block_dim_batch * block_dim_cin *
            block_dim_cout * block_dim_k * block_dim_group)
        fused_multi_core, block_idx = self._bind_core(blocks, block_dim_batch)

        def _split_w_for_conv1d():
            # the offset according to multicore
            hw_mad_1_l1_out_at = self.axis_dict.get("hw_mad_1_l1_out_at")
            hw_mad_1_l1_in_at = self.axis_dict.get("hw_mad_1_l1_in_at")
            if self.binary_mode:
                total_n = cin1_g * kernel_height * kernel_width
                total_m = cout_g // CUBE_DIM
                cache_tiling = self.binary_schedule.cache_tiling
                n_l0 = cache_tiling.get("n_ub_l0_time") * cache_tiling.get("cub_n1")
                m_l0 = cache_tiling.get("m_l0")
                if self.c_split_flag:
                    # for fp32 scene, n_l0 * n_single_core(even number) is equal to n_l0_n016 * n_single_core_n016
                    # because optiling use cin1_g is equal to cing / 8 in cache tiling
                    total_n = cube_util.shape_to_list(self.tensor_map.get("dw_ddr").shape)[1]
                block_dim_n = ceil(total_n, cache_tiling.get("n_single_core") * cache_tiling.get("n_bl1") * n_l0)
                block_dim_m = ceil(total_m, cache_tiling.get("m_single_core") * cache_tiling.get("m_al1") * m_l0)
                block_div = (ceil(batch_fmap, ceil(batch_fmap, block_dim_batch)) * block_dim_m *
                             block_dim_n * block_dim_group)
                hw_block_offset = block_idx // block_div * ceil(hw_pad_1, block_dim_k) * self.c0_size
            else:
                block_div = (ceil(batch_fmap, ceil(batch_fmap, block_dim_batch)) * block_dim_cout *
                             block_dim_cin * block_dim_group)
                hw_block_offset = fused_multi_core // block_div * ceil(hw_pad_1, block_dim_k) * self.c0_size
            if self.var_map:
                bool_dw_cc = dynamic_bl1_attach == "dw_cc"
            else:
                bool_dw_cc = tiling.get("BL1_shape") and (fmap_l1_tiling_nparts[0] != 1 or batch_num_sc != 1)
            if bool_dw_cc:
                if 'fmap_h' in self.var_map or 'fmap_w' in self.var_map:
                    bl1_k = tiling.get("BL1_shape")[0]
                    al1_k = grads_matrix_howo
                    if tiling.get("AL1_shape"):
                        al1_k = tiling.get("AL1_shape")[0]
                    if self.axis_dict.get("bl1_at_axis") == hw_mad_1_l1_in_at:
                        # hw splited two times before BL1 attach
                        hw_parts_offset = (hw_mad_1_l1_out_at.var * al1_k +
                                        hw_mad_1_l1_in_at.var * bl1_k)
                    else:
                        # hw splited one time before BL1  attach
                        hw_parts_offset = hw_mad_1_l1_out_at.var * bl1_k
                else:
                    hw_parts_offset = tvm.select(
                        tvm.all(grads_l1_tiling_nparts[0] > fmap_l1_tiling_nparts[0]),
                        # hw splited one time before BL1 attach
                        hw_mad_1_l1_out_at * tiling.get("BL1_shape")[0],
                        # hw splited 2 times before BL1 attach
                        ((hw_mad_1_l1_out_at * fmap_l1_tiling_nparts[0] // grads_l1_tiling_nparts[0]
                         + hw_mad_1_l1_in_at) * tiling.get("BL1_shape")[0]))

                hw_offset = hw_block_offset + hw_parts_offset
            else:
                # k axis is full load
                hw_offset = hw_block_offset
            # the offset of w according to that of k_axis
            hw_offset_with_pad = stride_width * hw_offset - pad_left
            # the extend of w according to that of k_axis
            if not tiling.get("BL1_shape"):
                kbl1_data = ceil(hw_pad_1 * self.c0_size, block_dim_k)
            else:
                kbl1_data = tiling.get("BL1_shape")[0]
            hw_extend = (kbl1_data - 1) * stride_width + kw_dilation
            if self.var_map:
                sch[self.tensor_map.get("fmap_l1")].buffer_tile(
                    (None, None), (None, None), (None, None),
                    (None, None), (hw_offset_with_pad, hw_extend), (None, None))
            else:
                sch[self.tensor_map.get("fmap_l1")].buffer_tile(
                    (None, None), (None, None), (None, None),
                    (hw_offset_with_pad, hw_extend), (None, None))

        def _tile_bound_bl1():
            if self.flag_load3d_w_split_case == 1 and not self.binary_mode:
                bl1_attach_mode = (fmap_l1_tiling_nparts[0] != 1 or batch_num_sc != 1
                                   or tiling.get("BL0_matrix")[0] * self.c0_size < width_grads)
                w_offset = None
                if bl1_attach_mode:
                    w_offset = self.axis_dict.get("wo_l0c_multi") * dw_k * self.c0_size * stride_width - pad_left
                w_extent = kw_dilation + (dw_k * self.c0_size - 1) * stride_width
                sch[self.tensor_map.get("fmap_l1")].buffer_tile(
                    (None, None), (None, None), (None, None),
                    (w_offset, w_extent), (None, None))
            elif self.flag_load3d_w_split_case == 1:
                return
            elif flag_conv1d_case:
                _split_w_for_conv1d()

        def _get_al1_bound():
            # al1 set storage bound
            if tiling.get("AL1_shape"):
                al1_m = tiling.get("AL1_shape")[1] * \
                              tiling.get("AL0_matrix")[0] * CUBE_DIM
                al1_k = tiling.get("AL1_shape")[0]
                al1_bound = al1_k * al1_m
            else:
                al1_m = grads_matrix_c1 * grads_matrix_c0
                al1_bound = grads_matrix_howo * al1_m
            return al1_bound

        def _dynamic_memory_management():
            # sequential_malloc
            sch.sequential_malloc(tbe_platform_info.scope_cbuf)
            sch.sequential_malloc(tbe_platform_info.scope_ca)
            sch.sequential_malloc(tbe_platform_info.scope_cb)
            sch.sequential_malloc(tbe_platform_info.scope_cc)
            if not self.support_l0c2out:
                sch.sequential_malloc(tbe_platform_info.scope_ubuf)

            # mem_unique
            sch[self.tensor_map.get("grads_matrix")].mem_unique()
            sch[self.tensor_map.get("fmap_matrix")].mem_unique()
            sch[self.tensor_map.get("grads_fractal")].mem_unique()
            sch[self.tensor_map.get("fmap_fractal")].mem_unique()
            sch[dw_cc].mem_unique()
            if not self.support_l0c2out:
                sch[dw_ub].mem_unique()

        def _manage_memory_process():
            if self.var_map:
                al1_bound = _get_al1_bound()
                bl1_bound = binary_schedule.cache_tiling.get("bl1_bound")
                # -1 is d_l0 index, d_l0 default value is 0
                al0_bound = reduce(lambda x, y: x * y, tiling.get("AL0_matrix")[:-1])
                bl0_bound = reduce(lambda x, y: x * y, tiling.get("BL0_matrix")[:-1])
                cl0_bound = reduce(lambda x, y: x * y, tiling.get("CL0_matrix")[:-1])
                sch[self.tensor_map.get("grads_matrix")].set_buffer_size(al1_bound)
                sch[self.tensor_map.get("fmap_matrix")].set_buffer_size(bl1_bound)
                sch[self.tensor_map.get("grads_fractal")].set_buffer_size(al0_bound)
                sch[self.tensor_map.get("fmap_fractal")].set_buffer_size(bl0_bound)
                sch[dw_cc].set_buffer_size(cl0_bound)
                _dynamic_memory_management()
            # when static fp32, grads matrix allocate may be nonliner, need to set buffer size
            elif self.c_split_flag:
                sch[self.tensor_map.get("grads_matrix")].set_buffer_size(_get_al1_bound())

        def _reverse_load_l0():
            # unsupport
            if self.binary_mode or dynamic_para or (
                    not self.reverse_params.get("reverse_load")) or (height_fmap * width_fmap) % c0_fmap != 0:
                return
            tensor_map_in_reverse = {
                "ddr": self.tensor_map.get("dw_ddr"),
                "l0c": dw_cc,
                "l1a": self.tensor_map.get("grads_matrix"),
                "l0a": self.tensor_map.get("grads_fractal"),
                "l1b": [self.tensor_map.get("fmap_matrix")],
                "l0b": self.tensor_map.get("fmap_fractal")
            }
            double_buffer_info = {
                "l0a": tiling.get("manual_pingpong_buffer").get("AL0_pbuffer"),
                "l0b": tiling.get("manual_pingpong_buffer").get("BL0_pbuffer")
            }
            if not flag_all_one_case:
                tensor_map_in_reverse.get("l1b").insert(0, self.tensor_map.get("fmap_l1"))

            self.reverse_params["axis_unit"] = conv2dBpFilterUtil.Conv2dbpFilterReverseLoad.get_axis_split_info(
                sch_agent, tensor_map_in_reverse)
            attach_info = {
                "l0a": {"stage": l0a_attach_scope, "axis": l0a_attach_axis},
                "l0b": {"stage": l0b_attach_scope, "axis": l0b_attach_axis},
                "l1a": {"stage": al1_attach_scope, "axis": al1_attach_axis},
                "l1b": {"stage": bl1_attach_scope, "axis": bl1_attach_axis},
                "l0c": {"stage": self.tensor_map.get("dw_ddr"), "axis": self.l0c_attach_axis}
            }
            conv2d_dw_reverse_load = conv2dBpFilterUtil.Conv2dbpFilterReverseLoad(
                sch, self.reverse_params.get("axis_unit"), attach_info, tensor_map_in_reverse, double_buffer_info)
            l0_at_axis_in_ddr = {
                "c_grads_mad_at": self.axis_dict.get("c_grads_mad_at"),
                "c_fmap_mad_at": self.axis_dict.get("c_fmap_mad_at")
            }
            l1_height_and_width_info = {
                "tiling_k": {"al1": tiling.get("AL1_shape")[0], "bl1": tiling.get("BL1_shape")[0]},
                "size": {"al1": (height_grads, width_grads), "bl1": (height_fmap, width_fmap)}
            }
            if_status = conv2d_dw_reverse_load.judge_exist_if([pad_left, pad_right, pad_up, pad_down],
                                                              l1_height_and_width_info,
                                                              [al1_attach_scope, bl1_attach_scope], dw_cc)
            self.reverse_params["reverse_status"] = conv2d_dw_reverse_load.reverse_load(l0_at_axis_in_ddr, if_status)
            if self.reverse_params.get("reverse_status"):
                self.reverse_params["control_reverse_axis"] = conv2d_dw_reverse_load.get_control_reverse_axis()
                self.reverse_params["reversed_tensor"] = conv2d_dw_reverse_load.get_reversed_tensor()
                self.reverse_params["k_axis_split_dict"] = conv2d_dw_reverse_load.get_split_k_axis_dict()

        _tile_bound_bl1()
        if self.l0b_dma_flag and not self.var_map:
            sch[self.tensor_map.get("fmap_l1")].compute_inline()
            sch[self.tensor_map.get("fmap_matrix")].compute_inline()
            self.tensor_map["fmap_l1"] = fmap_fractal_before
        l0a_attach_scope, l0a_attach_axis, l0b_attach_scope, l0b_attach_axis = _l0_attach()
        al1_attach_scope, al1_attach_axis = _al1_attach()
        bl1_attach_scope, bl1_attach_axis = _bl1_attach()
        # binary wout=1 scene need expand k, set shape_expand_time equal to 2
        shape_expand_time = get_load3d_special_factor(tiling.get("load3d_special_flag"))
        _split_binary_al1_process()
        _split_aub_process()
        _split_bub_process()
        _double_buffer()
        _handle_tbe_compile_para()
        _reverse_load_l0()
        _emit_insn()
        _set_bound_ub()
        _manage_memory_process()
        self.binary_schedule.binary_simplify(flag_conv1d_case, self.linear_embedding_opti_flag)

        return True

    def _dw_ddr_channel_split(self, factor_kw, factor_kh):
        """
        Split of fp32 input scene, channel_split feature
        """
        dw_ddr = self.tensor_map.get("dw_ddr")
        _, _, block_dim_cout, block_dim_cin, block_dim_group = self.tiling_para_dict.get("block_dim")
        # dw_shape is (real_g, fmap_channel_1*kernel_height*kernel_width,
        #              grads_channel, C0_fmap)
        g_multicore, g_axis = self.sch_agent[dw_ddr].split(self.sch[dw_ddr].op.axis[0], nparts=block_dim_group)

        c_fmap_multicore, c_fmap_mad_at = self.sch_agent[dw_ddr].split(self.sch[dw_ddr].op.axis[1],
                                                                       self.tiling_para_dict.get("n_single_core"))
        c_fmap_mad_at, c_fmap_mad_insn = self.sch_agent[dw_ddr].split(
            c_fmap_mad_at, nparts=self.tiling_para_dict.get("dw_tiling_nparts")[0])
        c_fmap_l1_ori, c_fmap_mad_at = self.sch_agent[dw_ddr].split(
            c_fmap_mad_at, nparts=self.tiling_para_dict.get("fmap_l1_tiling_nparts")[1])

        # split c_fmap_mad with factor=2 according to EmitInsn Channel_split
        c_fmap_mad_insn, c_fmap_mad_insn_inner = self.sch_agent[dw_ddr].split(c_fmap_mad_insn, factor=2)
        # split n dim
        c_fmap_l1_out, c_fmap_l1_at = self.sch_agent[dw_ddr].split(c_fmap_l1_ori, factor_kw)
        c_fmap_l1_c1, c_fmap_l1_kh = self.sch_agent[dw_ddr].split(c_fmap_l1_out, factor_kh)
        # split axis M, M axis located at 3th axis
        c_grads_mad_at, c_grads_mad_insn \
            = self.sch_agent[dw_ddr].split(self.sch[dw_ddr].op.axis[2],
                                           self.tiling_para_dict.get("dw_tiling_factor")[1]*CUBE_DIM)
        c_grads_multicore, c_grads_mad_at \
            = self.sch_agent[dw_ddr].split(c_grads_mad_at, nparts=block_dim_cout)
        c_grads_l1_at, c_grads_mad_at \
            = self.sch_agent[dw_ddr].split(c_grads_mad_at,
                                           nparts=self.tiling_para_dict.get("grads_l1_tiling_nparts")[1])
        # reorder according to requirments of mmad EmitInsn
        self.sch[dw_ddr].reorder(self.sch[dw_ddr].op.reduce_axis[0], g_multicore, c_grads_multicore, c_fmap_multicore,
                                 g_axis, c_fmap_l1_c1, c_fmap_l1_kh, c_fmap_l1_at, c_grads_l1_at, c_fmap_mad_at,
                                 c_grads_mad_at, c_fmap_mad_insn, c_fmap_mad_insn_inner, c_grads_mad_insn,
                                 self.sch[dw_ddr].op.axis[3])
        self.axis_dict["c_grads_mad_at"] = c_grads_mad_at
        self.axis_dict["c_grads_l1_at"] = c_grads_l1_at
        self.axis_dict["g_multicore"] = g_multicore
        self.axis_dict["c_grads_multicore"] = c_grads_multicore
        self.axis_dict["c_fmap_multicore"] = c_fmap_multicore
        self.axis_dict["c_fmap_l1_c1"] = c_fmap_l1_c1
        self.axis_dict["c_fmap_l1_kh"] = c_fmap_l1_kh
        self.axis_dict["c_fmap_l1_at"] = c_fmap_l1_at
        self.axis_dict["c_fmap_mad_at"] = c_fmap_mad_at
        self.axis_dict["c_fmap_mad_insn"] = c_fmap_mad_insn
        self.axis_dict["c_grads_mad_insn"] = c_grads_mad_insn

    def _dw_ddr_normal_split(self, factor_kw, factor_kh):
        """
        Split of common scenes
        """
        dw_ddr = self.tensor_map.get("dw_ddr")
        _, _, block_dim_cout, block_dim_cin, block_dim_group = self.tiling_para_dict.get("block_dim")
        # dw_shape is (real_g, fmap_channel_1*kernel_height*kernel_width,
        #              grads_channel_1, C0_grads, C0_fmap)
        g_multicore, g_axis = self.sch_agent[dw_ddr].split(self.sch[dw_ddr].op.axis[0],
                                                nparts=block_dim_group)
        c_fmap_multicore, c_fmap_mad_at \
            = self.sch_agent[dw_ddr].split(self.sch[dw_ddr].op.axis[1], self.tiling_para_dict.get("n_single_core"))
        c_fmap_mad_at, c_fmap_mad_insn \
            = self.sch_agent[dw_ddr].split(c_fmap_mad_at, nparts=self.tiling_para_dict.get("dw_tiling_nparts")[0])
        c_fmap_l1_ori, c_fmap_mad_at \
            = self.sch_agent[dw_ddr].split(c_fmap_mad_at,
                                           nparts=self.tiling_para_dict.get("fmap_l1_tiling_nparts")[1])
        # split n dim
        c_fmap_l1_out, c_fmap_l1_at = self.sch_agent[dw_ddr].split(c_fmap_l1_ori, factor_kw)
        c_fmap_l1_c1, c_fmap_l1_kh = self.sch_agent[dw_ddr].split(c_fmap_l1_out, factor_kh)
        # split axis M
        c_grads_mad_at, c_grads_mad_insn \
            = self.sch_agent[dw_ddr].split(self.sch[dw_ddr].op.axis[2],
                                           self.tiling_para_dict.get("dw_tiling_factor")[1]*CUBE_DIM)
        c_grads_multicore, c_grads_mad_at \
            = self.sch_agent[dw_ddr].split(c_grads_mad_at, nparts=block_dim_cout)
        c_grads_l1_at, c_grads_mad_at = \
            self.sch_agent[dw_ddr].split(c_grads_mad_at,
                                         nparts=self.tiling_para_dict.get("grads_l1_tiling_nparts")[1])
        if self.batch_al1 > 1:
            batch_ddr_factor = ceil(self.tiling_para_dict.get("batch_dim_factor"), self.batch_al1)
            batch_k_fuse_multicore, batch_ddr = self.sch_agent[dw_ddr].split(
                self.sch[dw_ddr].op.reduce_axis[0], factor=batch_ddr_factor)
            # reorder according to requirments of mmad EmitInsn
            self.sch[dw_ddr].reorder(batch_k_fuse_multicore,
                                 g_multicore, c_grads_multicore,
                                 c_fmap_multicore, batch_ddr, g_axis,
                                 c_fmap_l1_c1, c_fmap_l1_kh, c_fmap_l1_at,
                                 c_grads_l1_at,
                                 c_fmap_mad_at, c_grads_mad_at,
                                 c_fmap_mad_insn, c_grads_mad_insn, self.sch[dw_ddr].op.axis[3])
            self.axis_dict["batch_k_fuse_multicore"] = batch_k_fuse_multicore
            self.axis_dict["c_grads_multi_l1_at"] = batch_ddr
        else:
            self.sch[dw_ddr].reorder(self.sch[dw_ddr].op.reduce_axis[0],
                                    g_multicore, c_grads_multicore,
                                    c_fmap_multicore, g_axis,
                                    c_fmap_l1_c1, c_fmap_l1_kh, c_fmap_l1_at,
                                    c_grads_l1_at,
                                    c_fmap_mad_at, c_grads_mad_at,
                                    c_fmap_mad_insn, c_grads_mad_insn, self.sch[dw_ddr].op.axis[3])
        self.axis_dict["c_grads_mad_at"] = c_grads_mad_at
        self.axis_dict["c_grads_l1_at"] = c_grads_l1_at
        self.axis_dict["g_multicore"] = g_multicore
        self.axis_dict["c_grads_multicore"] = c_grads_multicore
        self.axis_dict["c_fmap_multicore"] = c_fmap_multicore
        self.axis_dict["c_fmap_l1_c1"] = c_fmap_l1_c1
        self.axis_dict["c_fmap_l1_kh"] = c_fmap_l1_kh
        self.axis_dict["c_fmap_l1_at"] = c_fmap_l1_at
        self.axis_dict["c_fmap_mad_at"] = c_fmap_mad_at
        self.axis_dict["c_fmap_mad_insn"] = c_fmap_mad_insn
        self.axis_dict["c_grads_mad_insn"] = c_grads_mad_insn

    def _dw_ddr_nhwc_split(self, factor_kw):
        """
        Split of NHWC output trans scenes
        """
        dw_ddr = self.tensor_map.get("dw_ddr")
        real_g = self.group_dict["real_g"].value
        _, _, block_dim_cout, block_dim_cin, _ = self.tiling_para_dict.get("block_dim")
        if not self.dw_fixpipe_flag:
            self.sch[self.tensor_map.get("dw_ddr")].compute_inline(instant=True)
            self.tensor_map["dw_ddr"] = self.tensor_map.get("dw_res_trans")
        dw_ddr = self.tensor_map.get("dw_ddr")
        ddr_batch, ddr_hw, ddr_c = self.sch[dw_ddr].op.axis
        # split the tensor axis to get [group, grads_c, hw, fmap_c1, fmap_c0]
        ddr_g, ddr_n = self.sch_agent[dw_ddr].split(ddr_batch, nparts=real_g)
        ddr_c1, ddr_c0 = self.sch_agent[dw_ddr].split(ddr_c, factor=16)
        # split multiple core axis
        g_multicore, g_axis = self.sch_agent[dw_ddr].split(ddr_g, nparts=real_g)
        # split n axis
        c_fmap_multicore, c_fmap_mad_at \
            = self.sch_agent[dw_ddr].split(ddr_c1, ceil(self.group_dict["cin1_g"], block_dim_cin))
        c_fmap_mad_at, c_fmap_mad_insn = \
            self.sch_agent[dw_ddr].split(c_fmap_mad_at, nparts=self.tiling_para_dict.get("dw_tiling_nparts")[0])
        c_fmap_l1_c1, c_fmap_mad_at \
            = self.sch_agent[dw_ddr].split(c_fmap_mad_at,
                                           nparts=self.tiling_para_dict.get("fmap_l1_tiling_nparts")[1])
        c_fmap_l1_kh, c_fmap_l1_at = self.sch_agent[dw_ddr].split(ddr_hw, factor_kw)
        # split m axis
        c_grads_mad_at, c_grads_mad_insn = \
            self.sch_agent[dw_ddr].split(ddr_n, self.tiling_para_dict.get("dw_tiling_factor")[1] * CUBE_DIM)
        c_grads_multicore, c_grads_mad_at = \
            self.sch_agent[dw_ddr].split(c_grads_mad_at, nparts=block_dim_cout)
        c_grads_l1_at, c_grads_mad_at \
            = self.sch_agent[dw_ddr].split(c_grads_mad_at,
                                           nparts=self.tiling_para_dict.get("grads_l1_tiling_nparts")[1])
        # reorder according to requirments of mmad EmitInsn
        self.sch[dw_ddr].reorder(self.sch[dw_ddr].op.reduce_axis[0],
                                 g_multicore,
                                 c_grads_multicore, c_fmap_multicore, g_axis,
                                 c_fmap_l1_c1, c_fmap_l1_kh, c_fmap_l1_at,
                                 c_grads_l1_at,
                                 c_fmap_mad_at, c_grads_mad_at,
                                 c_fmap_mad_insn, c_grads_mad_insn, ddr_c0)
        self.axis_dict["c_grads_mad_at"] = c_grads_mad_at
        self.axis_dict["c_grads_l1_at"] = c_grads_l1_at
        self.axis_dict["g_multicore"] = g_multicore
        self.axis_dict["c_grads_multicore"] = c_grads_multicore
        self.axis_dict["c_fmap_multicore"] = c_fmap_multicore
        self.axis_dict["c_fmap_l1_c1"] = c_fmap_l1_c1
        self.axis_dict["c_fmap_l1_kh"] = c_fmap_l1_kh
        self.axis_dict["c_fmap_l1_at"] = c_fmap_l1_at
        self.axis_dict["c_fmap_mad_at"] = c_fmap_mad_at
        self.axis_dict["c_fmap_mad_insn"] = c_fmap_mad_insn
        self.axis_dict["c_grads_mad_insn"] = c_grads_mad_insn

    def _dw_ddr_binary_split(self, kernel_width, kernel_height):
        """
        Split of binary scenes
        """
        dw_ddr = self.tensor_map.get("dw_ddr")
        _, _, block_dim_cout, block_dim_cin, block_dim_group = self.tiling_para_dict.get("block_dim")
        # dw_shape is (real_g, fmap_channel_1*kernel_height*kernel_width,
        #              grads_channel_1, C0_grads, C0_fmap)
        # for fp32 dw shape is (real_g, fmap_channel_1, kernel_height*kernel_width,
        #                       grads_channel, C0_fmap(8))
        g_multicore, g_axis = self.sch_agent[dw_ddr].split(self.sch[dw_ddr].op.axis[0],
                                                           nparts=block_dim_group,
                                                           split_params=self.binary_schedule.split_params)

        nparts_n_block_dim, nparts_n_l0, nparts_n_l1 = [None, None, None]
        factor_n_block_dim, factor_n_l0, factor_n_l1 = [None, None, None]
        nparts_m_block_dim, nparts_m_l0, nparts_m_l1 = [None, None, None]
        factor_m_block_dim, factor_m_l0, factor_m_l1 = [None, None, None]
        if self.binary_mode != cube_util.BinaryMode.NC1HWC0:
            nparts_n_block_dim = block_dim_cin
            nparts_n_l0 = self.tiling_para_dict.get("dw_tiling_nparts")[0]
            nparts_n_l1 = self.tiling_para_dict.get("fmap_l1_tiling_nparts")[1]

            nparts_m_block_dim = block_dim_cout
            nparts_m_l0 = self.tiling_para_dict.get("dw_tiling_nparts")[1]
            nparts_m_l1 = self.tiling_para_dict.get("grads_l1_tiling_nparts")[1]
            split_param0 = self.binary_schedule.split_params
        else:
            cache_tiling = self.binary_schedule.cache_tiling
            n_l0 = cache_tiling.get("n_ub_l0_time") * cache_tiling.get("cub_n1")
            # in FP32 scene, output n0 is 8 and the L0 n0 is 16, so n_single_core must be even number
            # n_single_core_fp32 already contains 2 x n_single_core_fp16
            factor_n_block_dim = cache_tiling.get("n_single_core") * cache_tiling.get("n_bl1") * n_l0
            factor_n_l0 = n_l0 * 2 if self.c_split_flag else n_l0
            factor_n_l1 = cache_tiling.get("n_bl1")

            factor_m_l0 = cache_tiling.get("m_l0") * CUBE_DIM
            factor_m_block_dim = cache_tiling.get("m_single_core") * cache_tiling.get("m_al1") * factor_m_l0
            factor_m_l1 = cache_tiling.get("m_al1")
            split_param0 = self.binary_schedule.split_nofactor_params

        c_fmap_multicore, c_fmap_mad_at \
            = self.sch_agent[dw_ddr].split(self.sch[dw_ddr].op.axis[1], factor=factor_n_block_dim,
                                           nparts=nparts_n_block_dim, split_params=split_param0)
        # factorization in ecah single core
        c_fmap_mad_at, c_fmap_mad_insn \
            = self.sch_agent[dw_ddr].split(c_fmap_mad_at, factor=factor_n_l0, nparts=nparts_n_l0,
                                           split_params=self.binary_schedule.split_params)
        c_fmap_axis_list = [c_fmap_mad_insn, ]
        if self.c_split_flag:
            c_fmap_mad_insn, c_fmap_mad_insn_inner = self.sch_agent[dw_ddr].split(c_fmap_mad_insn, 2)
            c_fmap_axis_list = [c_fmap_mad_insn, c_fmap_mad_insn_inner]
        c_fmap_l1_at, c_fmap_mad_at \
            = self.sch_agent[dw_ddr].split(c_fmap_mad_at, factor=factor_n_l1, nparts=nparts_n_l1,
                                           split_params=self.binary_schedule.split_params)

        # split axis M
        c_grads_multicore, c_grads_mad_at \
            = self.sch_agent[dw_ddr].split(self.sch[dw_ddr].op.axis[2], factor=factor_m_block_dim,
                                           nparts=nparts_m_block_dim, split_params=split_param0)
        c_grads_mad_at, c_grads_mad_insn \
            = self.sch_agent[dw_ddr].split(c_grads_mad_at, factor=factor_m_l0, nparts=nparts_m_l0,
                                           split_params=self.binary_schedule.split_params)
        c_grads_l1_at, c_grads_mad_at \
            = self.sch_agent[dw_ddr].split(c_grads_mad_at, factor=factor_m_l1, nparts=nparts_m_l1,
                                           split_params=self.binary_schedule.split_params)

        self.sch[dw_ddr].reorder(self.sch[dw_ddr].op.reduce_axis[0], g_multicore, c_grads_multicore,
                                 c_fmap_multicore, g_axis, c_fmap_l1_at, c_grads_l1_at, c_fmap_mad_at,
                                 c_grads_mad_at, *c_fmap_axis_list, c_grads_mad_insn, self.sch[dw_ddr].op.axis[3])
        self.axis_dict.update({
            "c_grads_mad_at": c_grads_mad_at, "c_grads_l1_at": c_grads_l1_at, "g_multicore": g_multicore,
            "c_grads_multicore": c_grads_multicore, "c_fmap_multicore": c_fmap_multicore, "c_fmap_l1_at": c_fmap_l1_at,
            "c_fmap_mad_at": c_fmap_mad_at, "c_fmap_mad_insn": c_fmap_mad_insn, "c_grads_mad_insn": c_grads_mad_insn})

    def _dw_ddr_split(self, kernel_width, kernel_height, factor_kw, factor_kh):
        """
        Split dw_ddr according to tiling
        """
        if self.binary_mode:
            self._dw_ddr_binary_split(kernel_width, kernel_height)
        elif self.c_split_flag and not self.dw_trans_flag:
            self._dw_ddr_channel_split(factor_kw, factor_kh)
        elif not self.dw_trans_flag:
            self._dw_ddr_normal_split(factor_kw, factor_kh)
        else:
            self._dw_ddr_nhwc_split(factor_kw)

    def _dw_cc_split_l0(self, dw_cc):
        """
        Split dw_cc according to L0 tiling
        """
        dw_k = self.tiling_para_dict.get("dw_k")
        axis_cc_outer, axis_cc_inner = None, None
        if self.flag_load3d_w_split_case == 1:
            wo_axis_single_core, batch_axis_single_core, ho_axis_single_core = self.sch[dw_cc].op.reduce_axis
            if self.var_map:
                k_l0_factor = dw_k * self.c0_size
            else:
                k_l0_factor = dw_k
                wo_axis_single_core, k_0 = self.sch_agent[dw_cc].split(wo_axis_single_core, self.c0_size)
            wo_l0c_multi, wo_l0c = self.sch_agent[dw_cc].split(wo_axis_single_core, k_l0_factor)
            batch_insn_o, batch_insn = self.sch_agent[dw_cc].split(batch_axis_single_core, 1)
            hw_mad_1_mad_at = ho_axis_single_core
            self.axis_dict["wo_l0c_multi"] = wo_l0c_multi
            axis_cc_outer = [batch_insn_o, wo_l0c_multi]
            axis_cc_inner = [wo_l0c, ] if self.var_map else [wo_l0c, k_0]
        else:
            if self.var_map:
                # get the 2 reduce axis of dw_cc
                batch_axis_single_core, k_1_axis_single_core = self.sch[dw_cc].op.reduce_axis
                k_l0_factor = dw_k * self.c0_size
            else:
                # get the 3 reduce axis of dw_cc
                batch_axis_single_core, k_1_axis_single_core, k_0 = self.sch[dw_cc].op.reduce_axis
                k_l0_factor = dw_k
            # dw_k is the part for one MMAD
            hw_mad_1_mad_at, hw_mad_1_mad_insn = self.sch_agent[dw_cc].split(k_1_axis_single_core, k_l0_factor)
            # mad_pattern :2 , the 1st axis should be 1, so do a fake split
            batch_insn_o, batch_insn = self.sch_agent[dw_cc].split(batch_axis_single_core, 1)
            axis_cc_outer = [batch_insn_o, ]
            axis_cc_inner = [hw_mad_1_mad_insn, ] if self.var_map else [hw_mad_1_mad_insn, k_0]
        self.axis_dict.update({"batch_insn_o": batch_insn_o, "batch_insn": batch_insn})
        return axis_cc_outer, axis_cc_inner, hw_mad_1_mad_at

    def _dw_cc_split(self, dw_cc, reduce_split_mode, load_order_change_flag):
        """
        Split dw_cc according to tiling
        """
        axis_cc_outer, axis_cc_inner, hw_mad_1_mad_at = self._dw_cc_split_l0(dw_cc)

        # K of AL1 and BL1 can be different, there are 2 split methods
        # on which one is larger
        if reduce_split_mode:
            hw_mad_1_l1_at, hw_mad_1_mad_at = self.sch_agent[dw_cc].split(
                hw_mad_1_mad_at, nparts=self.tiling_para_dict.get("grads_l1_tiling_nparts")[0])
            hw_mad_1_l1_out_at, hw_mad_1_l1_in_at = self.sch_agent[dw_cc].split(
                hw_mad_1_l1_at, nparts=self.tiling_para_dict.get("fmap_l1_tiling_nparts")[0])
            al1_at_axis = hw_mad_1_l1_in_at
            bl1_at_axis = hw_mad_1_l1_out_at
        else:
            hw_mad_1_l1_at, hw_mad_1_mad_at = self.sch_agent[dw_cc].split(
                hw_mad_1_mad_at, nparts=self.tiling_para_dict.get("fmap_l1_tiling_nparts")[0])
            hw_mad_1_l1_out_at, hw_mad_1_l1_in_at = self.sch_agent[dw_cc].split(
                hw_mad_1_l1_at, nparts=self.tiling_para_dict.get("grads_l1_tiling_nparts")[0])
            al1_at_axis = hw_mad_1_l1_out_at
            bl1_at_axis = hw_mad_1_l1_in_at
            if load_order_change_flag:
                al1_at_axis = hw_mad_1_l1_in_at
                bl1_at_axis = hw_mad_1_l1_out_at
        # split dw_cc.op.axis[0](N1), factor is one MMAD
        fkk_mad_at, fkk_mad_insn = self.sch_agent[dw_cc].split(
            self.sch[dw_cc].op.axis[2], self.tiling_para_dict.get("dw_tiling_factor")[0])

        # split dw_cc.op.axis[1](M1*M0), factor is one MMAD
        lc_mad_at, lc_mad_insn = self.sch_agent[dw_cc].split(
            self.sch[dw_cc].op.axis[3], self.tiling_para_dict.get("dw_tiling_factor")[1] * CUBE_DIM)

        batch_insn = self.axis_dict.get("batch_insn")
        if self.flag_load3d_w_split_case == 1:
            self.sch[dw_cc].reorder(self.sch[dw_cc].op.axis[1], fkk_mad_at, lc_mad_at, self.sch[dw_cc].op.axis[0],
                                *axis_cc_outer, hw_mad_1_l1_out_at,
                                hw_mad_1_l1_in_at, hw_mad_1_mad_at,
                                batch_insn, fkk_mad_insn, lc_mad_insn,
                                self.sch[dw_cc].op.axis[4], *axis_cc_inner)

        self.sch[dw_cc].reorder(fkk_mad_at, lc_mad_at, self.sch[dw_cc].op.axis[0],
                            *axis_cc_outer, hw_mad_1_l1_out_at,
                            hw_mad_1_l1_in_at, hw_mad_1_mad_at,
                            batch_insn, fkk_mad_insn, lc_mad_insn,
                            self.sch[dw_cc].op.axis[4], *axis_cc_inner)
        self.axis_dict.update({
                    "al1_at_axis": al1_at_axis, "bl1_at_axis": bl1_at_axis, "hw_mad_1_mad_at": hw_mad_1_mad_at,
                    "hw_mad_1_l1_out_at": hw_mad_1_l1_out_at, "hw_mad_1_l1_in_at": hw_mad_1_l1_in_at})

    def _bind_core_fuse_axis(self, blocks, block_dim_batch):
        """
        bind core by the fused axis
        """
        dw_ddr = self.tensor_map.get("dw_ddr")
        if self.batch_al1 > 1:
            fused_multi_core = \
                self.sch[dw_ddr].fuse(self.axis_dict.get("batch_k_fuse_multicore"), self.axis_dict.get("g_multicore"),
                                      self.axis_dict.get("c_grads_multicore"), self.axis_dict.get("c_fmap_multicore"))
        else:
            fused_multi_core = \
                self.sch[dw_ddr].fuse(self.sch[dw_ddr].op.reduce_axis[0], self.axis_dict.get("g_multicore"),
                                      self.axis_dict.get("c_grads_multicore"), self.axis_dict.get("c_fmap_multicore"))
        fused_multi_core, pragma_at = \
            self.sch[dw_ddr].split(fused_multi_core, 1)
        block = tvm.thread_axis("blockIdx.x")
        self.sch[dw_ddr].bind(fused_multi_core, block)

        if blocks == block_dim_batch:
            self.sch[dw_ddr].pragma(pragma_at, 'json_info_batchBindOnly')
        return fused_multi_core, block

    def _bind_core_axis_list(self):
        """
        bind core by the axis list
        """
        axis_list = [self.sch[self.tensor_map.get("dw_ddr")].op.reduce_axis[0], self.axis_dict.get("g_multicore"),
                     self.axis_dict.get("c_grads_multicore"), self.axis_dict.get("c_fmap_multicore")]
        block = tvm.thread_axis("blockIdx.x")
        self.sch.bind_axes(axis_list, block)
        self.sch[self.tensor_map.get("dw_ddr")].remove_init()
        return self.axis_dict.get("c_fmap_multicore"), block

    def _bind_core(self, blocks, block_dim_batch):
        """
        bind core according to tiling
        """
        if self.binary_mode:
            return self._bind_core_axis_list()

        return self._bind_core_fuse_axis(blocks, block_dim_batch)

    def _emit_insn_grads(self, flag_w_one_case, in_dtype):
        """
        Grads emit insn from ddr to L1 and L1 to L0A
        """
        # move grads from ddr to L1
        # when load3d special case, emit insn after H to avoid floor_div IR
        grads_ub2l1_idx = 3 if (flag_w_one_case and not self.binary_mode) else 0
        if self.grads_trans_flag:
            # grads_matrix axes(fp32): batch, mad_1, cout_1, mad_0->16, cout_0->8
            grads_matrix_emit_insn_axis = 2 if self.var_map else 0
            self.sch[self.tensor_map.get("grads")].compute_inline()
            self.sch[self.tensor_map.get("grads_matrix")].emit_insn(
                self.tensor_map.get("grads_matrix").op.axis[grads_matrix_emit_insn_axis],
                'dma_copy', {"layout_transform": "nd2nz"})
        else:
            if self.binary_mode == cube_util.BinaryMode.NC1HWC0:
                self.sch[self.tensor_map.get("grads_matrix")].pragma(
                    self.tensor_map.get("grads_matrix").op.axis[grads_ub2l1_idx], "loop_with_no_overlap_tensor")
            out2l1_insn_emit = "dma_padding" if in_dtype == "float32" else "dma_copy"
            self.sch[self.tensor_map.get("grads_matrix")].emit_insn(
                self.tensor_map.get("grads_matrix").op.axis[grads_ub2l1_idx], out2l1_insn_emit)
        # move grads from L1 to L0A
        grads_fractal = self.tensor_map.get("grads_fractal")
        grads_fractal_emit_axis = grads_fractal.op.axis[0]
        if in_dtype == "float32":
            self.sch[grads_fractal].split(grads_fractal.op.axis[-2], factor=self.c0_size)
            if not self.flag_load3d_w_split_case and self.binary_mode == cube_util.BinaryMode.NC1HWC0:
                self.sch[grads_fractal].pragma(grads_fractal_emit_axis, 'loop_with_no_overlap_tensor')
            if self.flag_load3d_w_split_case:
                self.sch[grads_fractal].emit_insn(grads_fractal_emit_axis, 'dma_copy', {'img2col': 1})
            else:
                # no_collapase: add label when axis_val is 1 for load3d for pass
                # w-splt: m-axis is above the producer in IR, the label no_collapase causes params error in load3d
                self.sch[grads_fractal].emit_insn(
                    grads_fractal_emit_axis, 'dma_padding', {'img2col': 1, 'no_collapase': 1})
        else:
            if self.binary_mode == cube_util.BinaryMode.NC1HWC0:
                self.sch[grads_fractal].pragma(grads_fractal_emit_axis, "loop_with_no_overlap_tensor")
            self.sch[grads_fractal].emit_insn(grads_fractal_emit_axis, 'dma_padding')

    def _emit_insn_fmap(self, flag_all_one_case, kernel_height, kernel_width, in_dtype):
        """
        Fmap emit insn from ddr to L1 and L1 to L0A
        """
        # move fmap from ddr to L1
        if flag_all_one_case:
            # load2d branch
            self._emit_insn_fmap_all_one_case(in_dtype)
        elif self.l0b_dma_flag:
            # dma brach
            self._emit_insn_fmap_l0b_dma()
        else:
            # load3d branch
            if self.var_map:
                self._emit_insn_fmap_dynamic(in_dtype)
            else:
                self._emit_insn_fmap_static()

    def _emit_insn_fmap_all_one_case(self, in_dtype):
        """
        Fmap emit insn for flag_all_one_case from ddr to L1 and L1 to L0A
        """
        if self.fmap_trans_flag:
            self.sch[self.tensor_map.get("fmap")].compute_inline()
            self.sch[self.tensor_map.get("fmap_matrix")].emit_insn(
                self.tensor_map.get("fmap_matrix").op.axis[2], 'dma_copy', {"layout_transform": "nd2nz"})
        else:
            self.sch[self.tensor_map.get("fmap_matrix")].pragma(
                self.tensor_map.get("fmap_matrix").op.axis[0], "loop_with_no_overlap_tensor")
            self.sch[self.tensor_map.get("fmap_matrix")].emit_insn(
                self.tensor_map.get("fmap_matrix").op.axis[0], 'dma_copy')
        # fmap_fractal emit_insn
        if in_dtype == "float32":
            self.sch[self.tensor_map.get("fmap_fractal")].split(
                self.tensor_map.get("fmap_fractal").op.axis[-2], factor=self.c0_size)
        fmap_fractal = self.tensor_map.get("fmap_fractal")
        emit_attr = {}
        if not self.fmap_trans_flag and self.binary_mode == cube_util.BinaryMode.NC1HWC0 \
            and self.binary_schedule.axis_merge_ins_flag == 0:
            # batch, ho*wo, ci1, hk, wk, c0
            axis_list = [fmap_fractal.op.axis[-4], fmap_fractal.op.axis[-5]]
            self._combine_cce_pragma(fmap_fractal, axis_list)
            emit_attr = {"axis_dynamic_shift": 1}
        self.sch[fmap_fractal].emit_insn(fmap_fractal.op.axis[0], 'dma_padding', emit_attr)

    def _emit_insn_fmap_l0b_dma(self):
        """
        Fmap emit insn for l0b_dma from ddr to L1 and L1 to L0A
        """
        emit_insn_tag_l1 = None
        if self.tensor_map.get("fmap_ub") is not None:
            # fmap_ub axes:
            #   batch, cin_1, fmap_h, fmap_w, cin_0->16
            # process 16 data at one time
            self.sch[self.tensor_map.get("fmap_ub")].emit_insn(
                self.tensor_map.get("fmap_ub").op.axis[4], "dma_padding")
            emit_insn_tag_l1 = "dma_copy"
        elif self.linear_embedding_opti_flag:
            emit_insn_tag_l1 = "dma_copy"
        else:
            emit_insn_tag_l1 = "dma_padding" if self.var_map else "dma_copy"
        # fmap_l1 axes:
        #   group, batch, hw_mad_1, fkk(cin_1*hk*wk), mad_0->16, cin_0->16
        self.sch[self.tensor_map.get("fmap_l1")].emit_insn(
            self.tensor_map.get("fmap_l1").op.axis[5], emit_insn_tag_l1)
        self.sch[self.tensor_map.get("fmap_l1")].pragma(
            self.tensor_map.get("fmap_l1").op.axis[4], "loop_with_no_overlap_tensor")
        if self.c_split_flag:
            self.sch[self.tensor_map.get("fmap_fractal")].split(
                self.tensor_map.get("fmap_fractal").op.axis[-2], factor=self.c0_size)
            self.sch[self.tensor_map.get("fmap_fractal")].emit_insn(
                self.tensor_map.get("fmap_fractal").op.axis[0], 'dma_copy', {'img2col': 1})
        else:
            self.sch[self.tensor_map.get("fmap_fractal")].emit_insn(
                self.tensor_map.get("fmap_fractal").op.axis[0], 'dma_copy')

    def _emit_insn_fmap_dynamic(self, in_dtype):
        """
        Fmap emit insn for dynamic from ddr to L1 and L1 to L0A
        """
        fmap_l1 = self.tensor_map.get("fmap_l1")
        fmap_fractal = self.tensor_map.get("fmap_fractal")
        fmap_l1_emit_insn_dict = self.setfmatrix_dict
        fmap_l1_emit_insn_axis = 0 if not self.fmap_trans_flag else -2
        if self.fmap_trans_flag:
            self.sch[self.tensor_map.get("fmap")].compute_inline()
            fmap_l1_emit_insn_dict["layout_transform"] = "nd2nz"
        if self.flag_load3d_w_split_case or in_dtype == "float32":
            # In the flag_load3d_w_split_case/fp32, both of AB need to load3d
            # So the A uses fmatrix, and the b uses fmatrixB
            # l1_target_scope equal 0 means L0A, and l1_target_scope equal 1 means L0B
            fmap_l1_emit_insn_dict["l1_target_scope"] = 1
        if self.binary_mode == cube_util.BinaryMode.NC1HWC0:
            if self.binary_schedule.bl1_attach_ins_flag == 1 and self.binary_schedule.axis_merge_ins_flag == 0:
                axis_list = [fmap_l1.op.axis[-2], fmap_l1.op.axis[-3]]
                self._combine_cce_pragma(fmap_l1, axis_list)
            self.sch[fmap_l1].pragma(fmap_l1.op.axis[fmap_l1_emit_insn_axis], "loop_with_no_overlap_tensor")
            self.sch[fmap_fractal].pragma(fmap_fractal.op.axis[0], "loop_with_no_overlap_tensor")

        self.sch[self.tensor_map.get("fmap_l1")].emit_insn(
            self.tensor_map.get("fmap_l1").op.axis[fmap_l1_emit_insn_axis], 'dma_copy', fmap_l1_emit_insn_dict)

        fmap_fractal_emit_insn_axis = self._get_emit_insn_fmap_fractal_axis()
        fmap_fractal_axis = self.sch[fmap_fractal].leaf_iter_vars
        cond = fmap_fractal_axis[0] > 0
        self.sch[fmap_fractal].set_value(cond, tvm.const(0.0, dtype=in_dtype), True)
        if in_dtype == "float32":
            # FP32(A load3d) need to use this tag, and it is likely to be unified in the future.
            # So the A uses fmatrix, and the b uses fmatrixB
            self.fmap_fractal_emit_insn_dict["fmatrix_ctrl"] = 1
        self.sch[fmap_fractal].emit_insn(fmap_fractal_emit_insn_axis, 'im2col_v2', self.fmap_fractal_emit_insn_dict)

    def _emit_insn_fmap_static(self):
        """
        Fmap emit insn for static from ddr to L1 and L1 to L0A
        """
        if self.fmap_trans_flag:
            self.sch[self.tensor_map.get("fmap_l1")].emit_insn(
                self.tensor_map.get("fmap_l1").op.axis[1], 'dma_copy', {"layout_transform": "nd2nz"})
        else:
            self.sch[self.tensor_map.get("fmap_l1")].emit_insn(
                self.tensor_map.get("fmap_l1").op.axis[0], 'dma_copy')

        if self.flag_load3d_w_split_case == 1:
            # fuse ho, wo axis for emit instruction
            _ = self.sch[self.tensor_map.get("fmap_matrix")].fuse(
                self.tensor_map.get("fmap_matrix").op.axis[1],
                self.tensor_map.get("fmap_matrix").op.axis[2])

        self.sch[self.tensor_map.get("fmap_matrix")].emit_insn(
            self.tensor_map.get("fmap_matrix").op.axis[0], 'set_fmatrix', self.setfmatrix_dict)
        self._emit_insn_fmap_fractal_normal()

    def _combine_cce_pragma(self, tensor, axis_list):
        """
        add pragma for emit_insn, in cachetiling scene, pass check whether can Combine multiple cce

        "axis_group": Merge continuous axis, reduce the number of tensor moves
        """
        id_val = tvm.call_extern("int32", "axis_group", 0, "overwrite")
        for axis in axis_list:
            self.sch[tensor].pragma(axis, "axis_group", id_val)

    def _emit_insn_fmap_fractal_normal(self):
        """
        fmap_fractal emit insn
        """
        fmap_fractal_emit_insn_axis = self._get_emit_insn_fmap_fractal_axis()
        self.sch[self.tensor_map.get("fmap_fractal")].emit_insn(fmap_fractal_emit_insn_axis, 'im2col')

    def _get_emit_insn_fmap_fractal_axis(self):
        """
        get fmap_fractal emit insn axis
        """
        if self.flag_load3d_w_split_case == 1:
            _, batch_axis, ho_axis, _, _, _, _ = self.tensor_map.get("fmap_fractal").op.axis
            self.sch[self.tensor_map.get("fmap_fractal")].reorder(ho_axis, batch_axis)
            return batch_axis
        return self.tensor_map.get("fmap_fractal").op.axis[1]

    def _show_flags(self):
        """
        show all flags
        """
        log.info("[{}] _c04_flag = {} _strideh_read_flag = {} l0b_dma_flag = {} c_split_flag = {}".format(
            self._op_name, self._c04_flag, self._strideh_read_flag, self.l0b_dma_flag, self.c_split_flag))
        log.info("[{}] dw_trans_flag = {} dw_fixpipe_flag = {} fmap_trans_flag = {} grads_trans_flag = {}".format(
            self._op_name, self.dw_trans_flag, self.dw_fixpipe_flag, self.fmap_trans_flag, self.grads_trans_flag))
        log.info("[{}] load3d_w_split_flag = {} linear_embedding_opti_flag = {}".format(
            self._op_name, self.flag_load3d_w_split_case, self.linear_embedding_opti_flag))


class Conv2DBpFilterBinaryDynamic(ConvBpFilterBinaryDynamic):
    """
    special for dynamic binary
    """
    def __init__(self, sch, binary_mode, flag_load3d_w_split_case, flag_all_one_case):
        super().__init__(sch, binary_mode, flag_load3d_w_split_case, flag_all_one_case)
