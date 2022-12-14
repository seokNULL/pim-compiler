# Copyright 2020 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for testing the relay pyxir frontend"""

import unittest
import numpy as np

from typing import List, Tuple

# ! To import tvm
import pyxir

try:
    import tvm
    from tvm import relay
    from tvm.relay import testing

    skip = False
except Exception as e:
    skip = True


if not skip:
    from pyxir.frontend.tvm import relay as xf_relay


class TestRelayL5VisionOperationConversions(unittest.TestCase):
    @unittest.skipIf(
        skip or not hasattr(relay.image, "resize"),
        "Could not import TVM and/or TVM frontend",
    )
    def test_image_resize_to_upsampling2d(self):
        # [DEPRECATED]
        data = relay.var("data", relay.TensorType((1, 20, 20, 32), "float32"))

        net = relay.image.resize(
            data,
            size=[40, 40],
            layout="NHWC",
            method="nearest_neighbor",
            coordinate_transformation_mode="asymmetric",
        )

        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)
        params = {}
        xgraph = xf_relay.from_relay(mod, params)
        layers = xgraph.get_layers()

        assert layers[1].type[0] == "Transpose"
        assert layers[1].shapes == [-1, 32, 20, 20]
        assert layers[2].type[0] == "Upsampling2D"
        assert layers[2].shapes == [-1, 32, 40, 40]
        assert layers[3].type[0] == "Transpose"
        assert layers[3].shapes == [-1, 40, 40, 32]

    @unittest.skipIf(
        skip or not hasattr(relay.image, "resize"),
        "Could not import TVM and/or TVM frontend",
    )
    def test_image_resize(self):
        # [DEPRECATED]
        data = relay.var("data", relay.TensorType((1, 20, 20, 32), "float32"))

        net = relay.image.resize(
            data,
            size=[40, 40],
            layout="NHWC",
            method="nearest_neighbor",
            coordinate_transformation_mode="half_pixel",
        )

        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)
        params = {}
        xgraph = xf_relay.from_relay(mod, params)
        layers = xgraph.get_layers()

        assert layers[1].type[0] == "AnyOp"
        assert layers[1].shapes == [-1, 40, 40, 32]

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_image_resize2d(self):
        def _test_image_resize2d(
            in_shape: Tuple[int],
            size: List[int],
            layout: str = "NHWC",
            method: str = "nearest_neighbor",
            coordinate_transformation_mode="asymmetric",
            rounding_method: str = "",
            cubic_alpha: float = -0.5,
            cubic_exclude: int = 0,
            supported: bool = True,
        ):
            n, h_in, w_in, c = [in_shape[layout.index(e)] for e in "NHWC"]
            h_out, w_out = size

            data = relay.var("data", relay.TensorType(in_shape, "float32"))
            net = relay.image.resize2d(
                data,
                size=size,
                layout=layout,
                method=method,
                coordinate_transformation_mode=coordinate_transformation_mode,
                rounding_method=rounding_method,
                cubic_alpha=cubic_alpha,
                cubic_exclude=cubic_exclude,
            )

            mod = tvm.IRModule.from_expr(net)
            mod = relay.transform.InferType()(mod)
            params = {}
            xgraph = xf_relay.from_relay(mod, params)
            layers = xgraph.get_layers()

            if supported:
                assert layers[1].type[0] == "Transpose"
                assert layers[1].shapes == [-1, c, h_in, w_in]
                assert layers[2].type[0] == "Upsampling2D"
                assert layers[2].shapes == [-1, c, h_out, w_out]
                assert layers[3].type[0] == "Transpose"
                assert layers[3].shapes == [-1, h_out, w_out, c]
            else:
                assert len(layers) == 2
                assert layers[1].type[0] == "AnyOp"
                assert layers[1].shapes == [-1, h_out, w_out, c]

        _test_image_resize2d((1, 20, 20, 32), [40, 40])
        _test_image_resize2d((1, 20, 20, 32), [10, 10])
        _test_image_resize2d((1, 15, 15, 32), [40, 40])

        # Unsupported
        _test_image_resize2d(
            (1, 20, 20, 32),
            [40, 40],
            coordinate_transformation_mode="half_pixel",
            supported=False,
        )
        _test_image_resize2d(
            (1, 20, 20, 32),
            [40, 40],
            coordinate_transformation_mode="align_corners",
            supported=False,
        )
        _test_image_resize2d(
            (1, 20, 20, 32), [40, 40], rounding_method="floor", supported=False
        )
        _test_image_resize2d(
            (1, 20, 20, 32), [40, 40], rounding_method="ceil", supported=False
        )
        _test_image_resize2d(
            (1, 20, 20, 32), [40, 40], rounding_method="round", supported=False
        )
        _test_image_resize2d((1, 20, 20, 32), [40, 40], cubic_alpha=-1, supported=False)
        _test_image_resize2d(
            (1, 20, 20, 32), [40, 40], cubic_exclude=1, supported=False
        )

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_yolo_reorg(self):
        data = relay.var("data", relay.TensorType((-1, 4, 2, 2), "float32"))

        net = relay.vision.yolo_reorg(data, stride=2)
        net = relay.Function(relay.analysis.free_vars(net), net)
        mod, params = testing.create_workload(net)

        xgraph = xf_relay.from_relay(mod, params)
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert layers[1].type[0] == "YoloReorg"

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_valid_counts(self):
        data = relay.var("data", relay.TensorType((-1, 2, 6), "float32"))

        net = relay.vision.get_valid_counts(data, score_threshold=1.0)[0]
        net = relay.Function(relay.analysis.free_vars(net), net)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert layers[1].type[0] == "Constant"
        assert layers[2].type[0] == "AnyOp"
        assert layers[2].shapes == [[1], [-1, 2, 6], [-1, 2]]

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_nms(self):
        data = relay.var("data", relay.TensorType((-1, 2, 6), "float32"))
        c = relay.expr.const(np.ones((1,), np.float32))
        indices = relay.var("indices", relay.TensorType((-1, 2), "float32"))

        net = relay.vision.non_max_suppression(data, c, indices)[0]
        net = relay.Function(relay.analysis.free_vars(net), net)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert layers[1].type[0] == "Constant"
        assert layers[2].type[0] == "Input"
        assert layers[3].type[0] == "Constant"
        assert layers[4].type[0] == "Constant"
        assert layers[5].type[0] == "AnyOp"
        assert layers[5].shapes == [[-1, 2], [-1, -1]]
