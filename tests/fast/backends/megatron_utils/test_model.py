import logging
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import Mock

import pytest
import torch


class FakeModelChunk:
    def __init__(self) -> None:
        self.zero_grad_buffer_count = 0

    def zero_grad_buffer(self) -> None:
        self.zero_grad_buffer_count += 1


class FakeDataIterator:
    def __init__(self) -> None:
        self.batches: list[dict[str, Any]] = []


class FakeMpu:
    def __init__(self, *, is_last_pipeline_stage: bool = False) -> None:
        self.is_last_pipeline_stage = is_last_pipeline_stage

    def is_pipeline_last_stage(self, ignore_virtual: bool = True, vp_stage: int | None = None) -> bool:
        return self.is_last_pipeline_stage


class FakeForwardBackwardEngine:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(kwargs)
        return []


@dataclass
class FakeParallelGroup:
    size: int = 1
    rank: int = 0


@dataclass
class FakeParallelState:
    indep_dp: FakeParallelGroup = field(default_factory=FakeParallelGroup)
    effective_dp: FakeParallelGroup = field(default_factory=FakeParallelGroup)


@dataclass
class TrainOneStepEnv:
    args: Namespace
    model: list[FakeModelChunk]
    data_iterator: list[FakeDataIterator]
    parallel_state: FakeParallelState
    forward_backward_engine: FakeForwardBackwardEngine


def make_train_one_step_args(**overrides: Any) -> Namespace:
    defaults: dict[str, Any] = dict(
        debug_disable_optimizer=False,
        multi_lora=False,
        custom_megatron_before_train_step_hook_path=None,
        dumper_enable=False,
        dumper_fwd_bwd=[],
        seq_length=8,
        decoder_seq_length=8,
        micro_batch_size=1,
        check_for_nan_in_loss_and_grad=True,
        ci_test=False,
        enable_mtp_training=False,
        rollout_max_response_len=512,
        use_sampling_support_replay=False,
        enable_witness=False,
        save_local_weight_checksum=False,
    )
    return Namespace(**{**defaults, **overrides})


@pytest.fixture
def train_one_step_env(monkeypatch) -> TrainOneStepEnv:
    from miles.backends.megatron_utils import model as model_module

    env = TrainOneStepEnv(
        args=make_train_one_step_args(),
        model=[FakeModelChunk()],
        data_iterator=[FakeDataIterator()],
        parallel_state=FakeParallelState(),
        forward_backward_engine=FakeForwardBackwardEngine(),
    )

    monkeypatch.setattr(model_module, "get_args", lambda: env.args)
    monkeypatch.setattr(model_module, "get_parallel_state", lambda: env.parallel_state)
    monkeypatch.setattr(model_module, "get_forward_backward_func", lambda: env.forward_backward_engine)
    monkeypatch.setattr(model_module, "mpu", FakeMpu())

    return env


class TestTrainOneStepStructuredLog:
    def test_train_one_step_emits_the_train_tag_in_its_structured_event(
        self, train_one_step_env: TrainOneStepEnv, caplog
    ):
        """Log consumers key train-step events off the train tag, so the caller must emit that tag with its fields."""
        from miles.backends.megatron_utils.model import train_one_step

        with caplog.at_level(logging.INFO, logger="miles.backends.megatron_utils.model"):
            train_one_step(
                args=train_one_step_env.args,
                rollout_id=7,
                step_id=3,
                data_iterator=train_one_step_env.data_iterator,
                model=train_one_step_env.model,
                optimizer=None,
                opt_param_scheduler=None,
                num_microbatches=1,
                num_rollouts=1,
                witness_info=None,
                attempt=2,
            )

        assert "train op=train_step rollout=7 step=3 attempt=2 outcome=NORMAL valid_step=true" in caplog.messages


def test_forward_only_omits_sampling_mask_for_callbacks_that_do_not_replay_sampling_support(monkeypatch):
    from miles.backends.megatron_utils import model as model_module

    def forward_backward(**kwargs):
        output, callback = kwargs["forward_step_func"](kwargs["data_iterator"][0], kwargs["model"][0])
        return [callback(output)]

    callback_kwargs = {}

    def critic_callback(_output, **kwargs):
        callback_kwargs.update(kwargs)
        return {}

    args = Namespace(
        data_pad_size_multiplier=1,
        qkv_format="thd",
        allgather_cp=False,
        enable_witness=False,
        use_rollout_entropy=False,
        custom_megatron_before_log_prob_hook_path=None,
        seq_length=8,
        micro_batch_size=1,
    )
    batch = {
        "tokens": None,
        "input_loss_masks": None,
        "multimodal_train_inputs": None,
        "unconcat_tokens": [],
        "total_lengths": [],
        "response_lengths": [],
        "max_seq_lens": None,
    }
    dumper = Mock()
    dumper.wrap_forward_step.side_effect = lambda fn: fn

    monkeypatch.setattr(model_module, "DumperMegatronUtil", lambda *_args, **_kwargs: dumper)
    monkeypatch.setattr(model_module, "get_model_config", lambda _model: Mock(timers=None))
    monkeypatch.setattr(model_module, "get_batch", lambda *_args, **_kwargs: batch)
    monkeypatch.setattr(model_module, "get_packed_seq_params", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(model_module, "get_forward_backward_func", lambda: forward_backward)
    monkeypatch.setattr(model_module, "mpu", FakeMpu())

    model_module.forward_only(
        critic_callback,
        args,
        [Mock()],
        [Mock()],
        [1],
        rollout_id=0,
    )

    assert "rollout_sampling_mask" not in callback_kwargs


def run_forward_only_step(monkeypatch, mpu, chunk) -> tuple[list[tuple[str, tuple[int, ...]]], torch.Tensor]:
    """Run forward_only over one microbatch of ``chunk`` under ``mpu``.

    Returns, in order, when the log-prob callback ran and what the forward step
    handed the schedule (by shape), plus the handed-over tensor itself.
    """
    from miles.backends.megatron_utils import model as model_module

    events = []
    returned = []

    def forward_backward(**kwargs):
        output, callback = kwargs["forward_step_func"](kwargs["data_iterator"][0], kwargs["model"][0])
        events.append(("returned", tuple(output.shape)))
        returned.append(output)
        return [callback(output, non_loss_data=True)]

    def collect(output, non_loss_data=True, **_kwargs):
        events.append(("collected", tuple(output.shape)))
        return {"log_probs": [output.sum()]}

    args = Namespace(
        data_pad_size_multiplier=1,
        qkv_format="thd",
        allgather_cp=False,
        enable_witness=False,
        use_rollout_entropy=False,
        custom_megatron_before_log_prob_hook_path=None,
        seq_length=8,
        micro_batch_size=1,
    )
    batch = {
        "tokens": None,
        "input_loss_masks": None,
        "multimodal_train_inputs": None,
        "unconcat_tokens": [],
        "total_lengths": [],
        "response_lengths": [],
        "max_seq_lens": None,
    }
    dumper = Mock()
    dumper.wrap_forward_step.side_effect = lambda fn: fn
    monkeypatch.setattr(model_module, "DumperMegatronUtil", lambda *_args, **_kwargs: dumper)
    monkeypatch.setattr(model_module, "get_model_config", lambda _model: Mock(timers=None))
    monkeypatch.setattr(model_module, "get_batch", lambda *_args, **_kwargs: batch)
    monkeypatch.setattr(model_module, "get_packed_seq_params", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(model_module, "get_forward_backward_func", lambda: forward_backward)
    monkeypatch.setattr(model_module, "mpu", mpu)
    monkeypatch.setattr(model_module, "aggregate_forward_results", lambda store, *_args, **_kwargs: store[0])

    model_module.forward_only(collect, args, [chunk], [Mock()], [1], rollout_id=0)
    return events, returned[0]


@pytest.mark.parametrize("last_stage", [True, False])
def test_forward_only_reduces_terminal_logits_inside_the_forward_step(monkeypatch, last_stage):
    """The last stage collects log-probs before returning, so the pipeline never holds two logits buffers."""
    chunk = Mock(return_value=torch.ones(1, 3, 7))

    events, _ = run_forward_only_step(monkeypatch, FakeMpu(is_last_pipeline_stage=last_stage), chunk)

    if last_stage:
        assert events == [("collected", (1, 3, 7)), ("returned", ())]
    else:
        assert events == [("returned", (1, 3, 7)), ("collected", (1, 3, 7))]


@pytest.mark.parametrize("vp_stage", [0, 1])
def test_forward_only_reduces_only_on_the_final_virtual_stage(monkeypatch, vp_stage):
    """With VPP 2 on the last pipeline rank, both chunks pass Megatron's default
    is_pipeline_last_stage(). Only virtual stage 1 outputs logits; stage 0's hidden
    states must reach the next virtual stage untouched, not as a reduced placeholder."""
    from megatron.core import parallel_state

    monkeypatch.setattr(parallel_state, "_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE", 2)
    monkeypatch.setattr(parallel_state, "_MPU_PIPELINE_MODEL_PARALLEL_RANK", 1)
    monkeypatch.setattr(parallel_state, "_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE", 2)
    output = torch.ones(1, 3, 7)

    events, returned = run_forward_only_step(monkeypatch, parallel_state, Mock(return_value=output, vp_stage=vp_stage))

    if vp_stage == 1:
        assert events == [("collected", (1, 3, 7)), ("returned", ())]
    else:
        assert returned is output
        assert events == [("returned", (1, 3, 7)), ("collected", (1, 3, 7))]
