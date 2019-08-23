# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.distributed as dist

from thumt.optimizers.schedules import LearningRateSchedule


class Optimizer(object):

    def __init__(self, name, **kwargs):
        self._name = name
        self._iterations = 0
        self._slots = {}

    def detach_gradients(self, gradients):
        for grad in gradients:
            if grad is not None:
                grad.detach_()

    def scale_gradients(self, gradients, scale):
        for grad in gradients:
            if grad is not None:
                grad.mul_(scale)

    def sync_gradients(self, gradients, compress=True):
        for grad in gradients:
            if grad is None:
                continue

            if compress and grad.dtype != torch.float16:
                grad_fp16 = grad.half()
                dist.all_reduce(grad_fp16)
                grad.copy_(grad_fp16)
            else:
                dist.all_reduce(grad)

    def zero_gradients(self, gradients):
        for grad in gradients:
            if grad is not None:
                grad.zero_()

    def compute_gradients(self, loss, var_list, aggregate=False):
        var_list = list(var_list)
        grads = [v.grad if v is not None else None for v in var_list]

        self.detach_gradients(grads)

        if not aggregate:
            self.zero_gradients(grads)

        loss.backward()
        return [v.grad if v is not None else None for v in var_list]

    def apply_gradients(self, grads_and_vars):
        raise NotImplementedError("Not implemented")

    @property
    def iterations(self):
        return self._iterations

    def state_dict(self):
        raise NotImplementedError("Not implemented")

    def load_state_dict(self):
        raise NotImplementedError("Not implemented")


class AdamOptimizer(Optimizer):

    def __init__(self, learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-7, name="Adam", **kwargs):
        super(AdamOptimizer, self).__init__(name, **kwargs)
        self._learning_rate = learning_rate
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._epsilon = epsilon

    def apply_gradients(self, grads_and_vars):
        self._iterations += 1
        lr = self._learning_rate
        beta_1 = self._beta_1
        beta_2 = self._beta_2
        epsilon = self._epsilon

        for grad, var in grads_and_vars:
            if grad is None:
                continue

            # Convert if grad is not FP32
            grad = grad.data.float()
            name, var = var

            if self._slots.get(name, None) is None:
                self._slots[name] = {}
                self._slots[name]["m"] = torch.zeros_like(var.data,
                                                          dtype=torch.float32)
                self._slots[name]["v"] = torch.zeros_like(var.data,
                                                          dtype=torch.float32)

            m, v = self._slots[name]["m"], self._slots[name]["v"]

            bias_corr_1 = 1 - beta_1 ** self._iterations
            bias_corr_2 = 1 - beta_2 ** self._iterations

            m.mul_(beta_1).add_(1 - beta_1, grad)
            v.mul_(beta_2).addcmul_(1 - beta_2, grad, grad)
            denom = (v.sqrt() / math.sqrt(bias_corr_2)).add_(epsilon)

            if isinstance(lr, LearningRateSchedule):
                lr = lr(self._iterations)

            step_size = lr / bias_corr_1

            if var.dtype == torch.float32:
                var.data.addcdiv_(-step_size, m, denom)
            else:
                fp32_var = var.data.float()
                fp32_var.addcdiv_(-step_size, m, denom)
                var.data.copy_(fp32_var)

    def state_dict(self):
        state = {
            "beta_1": self._beta_1,
            "beta_2": self._beta_2,
            "epsilon": self._epsilon,
            "iterations": self._iterations,
            "slot": self._slots
        }

        if not isinstance(self._learning_rate, LearningRateSchedule):
            state["learning_rate"] = self._learning_rate

        return state

    def load_state_dict(self, state):
        self._learning = state.get("learning_rate", self._learning_rate)
        self._beta_1 = state.get("beta_1", self._beta_1)
        self._beta_2 = state.get("beta_2", self._beta_2)
        self._epsilon = state.get("epsilon", self._epsilon)
        self._iterations = state.get("iterations", self._iterations)

        slots = state.get("slots", {})
        self._slots = {}

        for key in slots:
            m, v = slots[key]["m"], slots[key]["v"]
            self._slots[key]["m"] = torch.zeros(m.shape, dtype=torch.float32)
            self._slots[key]["v"] = torch.zeros(v.shape, dtype=torch.float32)
            self._slots[key]["m"].copy_(m)
            self._slots[key]["v"].copy_(v)


class LossScalingOptimizer(Optimizer):

    def __init__(self, optimizer, scale=2.0**7, increment_period=2000,
                 multiplier=2.0, name="LossScalingOptimizer", **kwargs):
        super(LossScalingOptimizer, self).__init__(name, **kwargs)
        self._optimizer = optimizer
        self._scale = scale
        self._increment_period = increment_period
        self._multiplier = multiplier
        self._num_good_steps = 0

    def _update_if_finite_grads(self):
        if self._num_good_steps + 1 > self._increment_period:
            self._scale *= self._multiplier
            self._num_good_steps = 0
        else:
            self._num_good_steps += 1

    def _update_if_not_finite_grads(self):
        self._scale = max(self._scale / self._multiplier, 1)

    def compute_gradients(self, loss, var_list, aggregate=False):
        var_list = list(var_list)
        grads = [v.grad if v is not None else None for v in var_list]

        self.detach_gradients(grads)

        if not aggregate:
            self.zero_gradients(grads)

        loss = loss * self._scale
        loss.backward()

        return [v.grad if v is not None else None for v in var_list]

    def apply_gradients(self, grads_and_vars):
        grads, var_list = list(zip(*grads_and_vars))
        new_grads = []

        for grad in grads:
            if grad is None:
                new_grads.append(None)
                continue

            norm = grad.data.norm()

            if not torch.isfinite(norm):
                self._update_if_not_finite_grads()
                return
            else:
                # Rescale gradients
                new_grads.append(grad.data.float().mul_(1.0 / self._scale))

        self._update_if_finite_grads()
        self._optimizer.apply_gradients(zip(new_grads, var_list))

    def state_dict(self):
        state = {
            "scale": self._scale,
            "increment_eriod": self._increment_period,
            "multiplier": self._multiplier,
            "num_good_steps": self._num_good_steps,
            "optimizer": self._optimizer.state_dict()
        }
        return state

    def load_state_dict(self, state):
        self._scale = state.get("scale", self._scale)
        self._increment_period = state.get("increment_period",
                                           self._increment_period)
        self._multiplier = state.get("multiplier", self._multiplier)
        self._num_good_steps = state.get("num_good_steps",
                                         self._num_good_steps)
        self._optimizer.load_state_dict(state.get("optimizer", {}))


class MultiStepOptimizer(Optimizer):

    def __init__(self, optimizer, n=1, compress=True,
                 name="MultiStepOptimizer", **kwargs):
        super(MultiStepOptimizer, self).__init__(name, **kwargs)
        self._n = n
        self._optimizer = optimizer
        self._compress = compress

    def compute_gradients(self, loss, var_list, aggregate=False):
        if self._iterations % self._n == 0:
            return self._optimizer.compute_gradients(loss, var_list, aggregate)
        else:
            return self._optimizer.compute_gradients(loss, var_list, True)

    def apply_gradients(self, grads_and_vars):
        size = dist.get_world_size()
        grads, var_list = list(zip(*grads_and_vars))
        self._iterations += 1

        if self._n == 1:
            if size > 1:
                self.sync_gradients(grads, compress=self._compress)
                self.scale_gradients(grads, 1.0 / size)

            self._optimizer.apply_gradients(zip(grads, var_list))
        else:
            if self._iterations % self._n != 0:
                return

            if size > 1:
                self.sync_gradients(grads, compress=self._compress)

            self.scale_gradients(grads, 1.0 / (self._n * size))
            self._optimizer.apply_gradients(zip(grads, var_list))

    def state_dict(self):
        state = {
            "n": self._n,
            "iterations": self._iterations,
            "compress": self._compress,
            "optimizer": self._optimizer.state_dict()
        }
        return state

    def load_state_dict(self, state):
        self._n = state.get("n", self._n)
        self._iterations = state.get("iterations", self._iterations)
        self._compress = state.get("compress", self._iterations)
        self._optimizer.load_state_dict(state.get("optimizer", {}))