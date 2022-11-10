import time
import torch
import unittest
from src.utils import *


class MonolithicPerformance(unittest.TestCase):
    def initialize(self):
        self.repeats = 1000
        self.box_list = [
            [int(x) for x in torch.randint(0, 500, (1, 4))[0].sort().values]
            for i in range(20)
        ]
        self.box_list = BoxList(self.box_list, (720, 1280))
        self.box_list.extra_fields["labels"] = torch.randint(0, 10, (200, 1))
        self.strides = torch.tensor([8, 16, 32, 64, 128])

    def step1_criterium(self):
        start = time.perf_counter()
        for i in range(self.repeats):
            self.box_target_stride = criterium(
                self.box_list, self.strides, object_sizes_of_interest
            )
        stop = time.perf_counter()
        return stop - start

    def step2_get_cls_target(self):
        start = time.perf_counter()
        for i in range(self.repeats):
            self.maps_cls = get_cls_target(
                self.box_list,
                self.strides,
                self.box_target_stride,
                self.box_list.extra_fields["labels"],
                "cpu",
            )
        stop = time.perf_counter()
        return stop - start

    def step3_get_reg_target(self):
        start = time.perf_counter()
        for i in range(self.repeats):
            self.maps_reg = get_reg_target(
                self.box_list, self.strides, self.box_target_stride, "cpu"
            )
        stop = time.perf_counter()
        return stop - start

    def step4_get_cnt_target(self):
        start = time.perf_counter()
        for i in range(self.repeats):
            self.maps_cnt = get_cnt_target(
                self.box_list, self.strides, self.maps_reg, "cpu"
            )
        stop = time.perf_counter()
        return stop - start

    def _steps(self):
        for name in dir(self):  # dir() result is implicitly sorted
            if name.startswith("step"):
                yield name, getattr(self, name)

    def test_steps(self):
        self.initialize()
        for name, step in self._steps():
            try:
                execution_time = step()
                print(
                    f"{name:<20} time:{execution_time :.3f} step time: {execution_time/300 :5f}"
                )
            except Exception as e:
                self.fail("{} failed ({}: {})".format(step, type(e), e))
