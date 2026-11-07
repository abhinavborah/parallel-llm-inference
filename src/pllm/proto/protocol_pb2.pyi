from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class RegisterReq(_message.Message):
    __slots__ = ("stage", "layer_start", "layer_end", "model_id", "weights_sha256", "addr")
    STAGE_FIELD_NUMBER: _ClassVar[int]
    LAYER_START_FIELD_NUMBER: _ClassVar[int]
    LAYER_END_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_SHA256_FIELD_NUMBER: _ClassVar[int]
    ADDR_FIELD_NUMBER: _ClassVar[int]
    stage: int
    layer_start: int
    layer_end: int
    model_id: str
    weights_sha256: bytes
    addr: str
    def __init__(self, stage: _Optional[int] = ..., layer_start: _Optional[int] = ..., layer_end: _Optional[int] = ..., model_id: _Optional[str] = ..., weights_sha256: _Optional[bytes] = ..., addr: _Optional[str] = ...) -> None: ...

class RegisterResp(_message.Message):
    __slots__ = ("epoch",)
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    epoch: int
    def __init__(self, epoch: _Optional[int] = ...) -> None: ...

class Beat(_message.Message):
    __slots__ = ("stage", "epoch", "inbox_depth", "cached_requests")
    STAGE_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    INBOX_DEPTH_FIELD_NUMBER: _ClassVar[int]
    CACHED_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    stage: int
    epoch: int
    inbox_depth: int
    cached_requests: int
    def __init__(self, stage: _Optional[int] = ..., epoch: _Optional[int] = ..., inbox_depth: _Optional[int] = ..., cached_requests: _Optional[int] = ...) -> None: ...

class Nack(_message.Message):
    __slots__ = ("stage", "epoch", "rid", "attempt", "step", "reason")
    STAGE_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    RID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    stage: int
    epoch: int
    rid: str
    attempt: int
    step: int
    reason: str
    def __init__(self, stage: _Optional[int] = ..., epoch: _Optional[int] = ..., rid: _Optional[str] = ..., attempt: _Optional[int] = ..., step: _Optional[int] = ..., reason: _Optional[str] = ...) -> None: ...

class StageMsg(_message.Message):
    __slots__ = ("beat", "nack")
    BEAT_FIELD_NUMBER: _ClassVar[int]
    NACK_FIELD_NUMBER: _ClassVar[int]
    beat: Beat
    nack: Nack
    def __init__(self, beat: _Optional[_Union[Beat, _Mapping]] = ..., nack: _Optional[_Union[Nack, _Mapping]] = ...) -> None: ...

class RouteUpdate(_message.Message):
    __slots__ = ("next_stage", "addr", "epoch")
    NEXT_STAGE_FIELD_NUMBER: _ClassVar[int]
    ADDR_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    next_stage: int
    addr: str
    epoch: int
    def __init__(self, next_stage: _Optional[int] = ..., addr: _Optional[str] = ..., epoch: _Optional[int] = ...) -> None: ...

class Free(_message.Message):
    __slots__ = ("rid", "attempt")
    RID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    rid: str
    attempt: int
    def __init__(self, rid: _Optional[str] = ..., attempt: _Optional[int] = ...) -> None: ...

class ControlMsg(_message.Message):
    __slots__ = ("route_update", "free")
    ROUTE_UPDATE_FIELD_NUMBER: _ClassVar[int]
    FREE_FIELD_NUMBER: _ClassVar[int]
    route_update: RouteUpdate
    free: Free
    def __init__(self, route_update: _Optional[_Union[RouteUpdate, _Mapping]] = ..., free: _Optional[_Union[Free, _Mapping]] = ...) -> None: ...

class Tensor(_message.Message):
    __slots__ = ("dtype", "shape", "data")
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    dtype: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    data: bytes
    def __init__(self, dtype: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ..., data: _Optional[bytes] = ...) -> None: ...

class TokenIds(_message.Message):
    __slots__ = ("ids",)
    IDS_FIELD_NUMBER: _ClassVar[int]
    ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, ids: _Optional[_Iterable[int]] = ...) -> None: ...

class StepRequest(_message.Message):
    __slots__ = ("rid", "attempt", "step", "start_pos", "length", "target_epoch", "sender_epoch", "replay", "deadline_unix_ms", "tokens", "activations")
    RID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    START_POS_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    TARGET_EPOCH_FIELD_NUMBER: _ClassVar[int]
    SENDER_EPOCH_FIELD_NUMBER: _ClassVar[int]
    REPLAY_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    ACTIVATIONS_FIELD_NUMBER: _ClassVar[int]
    rid: str
    attempt: int
    step: int
    start_pos: int
    length: int
    target_epoch: int
    sender_epoch: int
    replay: bool
    deadline_unix_ms: int
    tokens: TokenIds
    activations: Tensor
    def __init__(self, rid: _Optional[str] = ..., attempt: _Optional[int] = ..., step: _Optional[int] = ..., start_pos: _Optional[int] = ..., length: _Optional[int] = ..., target_epoch: _Optional[int] = ..., sender_epoch: _Optional[int] = ..., replay: _Optional[bool] = ..., deadline_unix_ms: _Optional[int] = ..., tokens: _Optional[_Union[TokenIds, _Mapping]] = ..., activations: _Optional[_Union[Tensor, _Mapping]] = ...) -> None: ...

class StepResult(_message.Message):
    __slots__ = ("rid", "attempt", "step", "epoch", "token", "top_logit")
    RID_FIELD_NUMBER: _ClassVar[int]
    ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    TOP_LOGIT_FIELD_NUMBER: _ClassVar[int]
    rid: str
    attempt: int
    step: int
    epoch: int
    token: int
    top_logit: float
    def __init__(self, rid: _Optional[str] = ..., attempt: _Optional[int] = ..., step: _Optional[int] = ..., epoch: _Optional[int] = ..., token: _Optional[int] = ..., top_logit: _Optional[float] = ...) -> None: ...

class GenerateReq(_message.Message):
    __slots__ = ("prompt", "max_tokens", "timeout_ms")
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
    prompt: str
    max_tokens: int
    timeout_ms: int
    def __init__(self, prompt: _Optional[str] = ..., max_tokens: _Optional[int] = ..., timeout_ms: _Optional[int] = ...) -> None: ...

class Token(_message.Message):
    __slots__ = ("rid", "step", "token", "text", "top_logit")
    RID_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    TOP_LOGIT_FIELD_NUMBER: _ClassVar[int]
    rid: str
    step: int
    token: int
    text: str
    top_logit: float
    def __init__(self, rid: _Optional[str] = ..., step: _Optional[int] = ..., token: _Optional[int] = ..., text: _Optional[str] = ..., top_logit: _Optional[float] = ...) -> None: ...
