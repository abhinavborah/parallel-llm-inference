"""Failure detection at the gateway (R6.3, R6.4): the detector on a fake clock."""

import json
import logging
import threading

from pllm.detection import MAX_EVENTS, Detector

SECOND = 1_000_000_000
STAGE, EPOCH = 1, 7


class Clock:
    def __init__(self):
        self.ns = 1000

    def __call__(self):
        return self.ns

    def advance(self, seconds):
        self.ns += round(seconds * SECOND)


def fake(**kwargs):
    clock = Clock()
    return Detector(clock=clock, wall=lambda: 1234.5, **kwargs), clock


def detected(detector, **fields):
    return [
        event
        for event in detector.events()
        if event["event"] == "detected" and fields.items() <= event.items()
    ]


def test_a_registered_stage_that_never_beat_is_never_expired():
    detector, clock = fake()
    clock.advance(3600)
    detector.expire()
    assert detector.events() == []


def test_a_silent_stage_expires_at_the_timeout_and_not_before():
    detector, clock = fake(heartbeat_timeout=2.0)
    detector.beat(STAGE, EPOCH)
    beat_at = clock.ns
    clock.advance(1.999)
    detector.expire()
    assert detector.events() == [] and not detector.is_dead(EPOCH)
    clock.advance(0.001)
    detector.expire()
    assert detected(detector) == [
        {
            "event": "detected",
            "path": "heartbeat_timeout",
            "stage": STAGE,
            "epoch": EPOCH,
            "declared": True,
            "t_ns": clock.ns,
            "time": 1234.5,
            "last_beat_ns": beat_at,
            "since_last_beat_ns": 2 * SECOND,
            "rid": None,
            "step": None,
            "detail": "",
        }
    ]
    assert detector.is_dead(EPOCH)


def test_a_beat_keeps_a_stage_alive():
    detector, clock = fake(heartbeat_timeout=2.0)
    for _ in range(10):
        detector.beat(STAGE, EPOCH)
        clock.advance(1.5)
        detector.expire()
    assert detector.events() == []


def test_a_new_epoch_does_not_inherit_the_last_beat_of_the_old_one():
    detector, clock = fake(heartbeat_timeout=2.0)
    detector.beat(STAGE, EPOCH)
    detector.declare(STAGE, EPOCH, "stream_break")
    clock.advance(1.5)
    detector.beat(STAGE, EPOCH + 1)
    clock.advance(1.0)
    detector.expire()
    assert not detector.is_dead(EPOCH + 1)
    assert len(detected(detector)) == 1


def test_each_epoch_of_a_stage_has_its_own_last_beat():
    detector, clock = fake(heartbeat_timeout=2.0)
    detector.beat(STAGE, EPOCH)
    clock.advance(1.5)
    detector.beat(STAGE, EPOCH + 1)
    clock.advance(1.0)
    detector.expire()
    assert detector.is_dead(EPOCH) and not detector.is_dead(EPOCH + 1)


def test_a_dead_epoch_is_never_revived_by_a_late_beat():
    detector, clock = fake(heartbeat_timeout=2.0)
    detector.beat(STAGE, EPOCH)
    detector.declare(STAGE, EPOCH, "stream_break")
    assert detector.beat(STAGE, EPOCH) is False
    clock.advance(5)
    detector.expire()
    assert len(detected(detector)) == 1 and detector.is_dead(EPOCH)


def test_the_first_declaration_wins_from_many_threads():
    detector, _ = fake()
    detector.beat(STAGE, EPOCH)
    start = threading.Barrier(16)
    wins = []

    def declare(path):
        start.wait()
        wins.append(detector.declare(STAGE, EPOCH, path))

    threads = [threading.Thread(target=declare, args=(f"path{i}",)) for i in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(wins) == [False] * 15 + [True]
    assert len(detected(detector)) == 1


def test_a_step_deadline_declares_nothing_and_has_no_suspect_while_beats_are_fresh():
    detector, clock = fake(heartbeat_timeout=2.0)
    detector.beat(STAGE, EPOCH)
    clock.advance(1.0)  # exactly half: not yet suspect
    detector.step_deadline("r1", 4)
    record = detected(detector, path="step_deadline")[0]
    assert (record["rid"], record["step"], record["declared"]) == ("r1", 4, False)
    assert record["stage"] is None and record["epoch"] is None
    assert record["last_beat_ns"] is None and record["since_last_beat_ns"] is None
    assert not detector.is_dead(EPOCH)


def test_a_step_deadline_names_the_stage_with_the_oldest_beat_as_suspect():
    detector, clock = fake(heartbeat_timeout=2.0)
    detector.beat(0, 3)
    clock.advance(0.5)
    detector.beat(1, 4)
    clock.advance(1.0)  # stage 0 last beat 1.5 s ago, stage 1 1.0 s ago
    detector.step_deadline("r1", 0)
    record = detected(detector, path="step_deadline")[0]
    assert (record["stage"], record["epoch"]) == (0, 3)
    assert record["since_last_beat_ns"] == 1.5 * SECOND
    assert not detector.is_dead(3)


def test_a_registration_reports_the_wait_since_the_declaration():
    detector, clock = fake()
    detector.declare(STAGE, EPOCH, "stream_break")
    clock.advance(3)
    detector.registered(STAGE, EPOCH, EPOCH + 1, clock.ns)
    assert detector.events()[-1] == {
        "event": "registered",
        "stage": STAGE,
        "old_epoch": EPOCH,
        "new_epoch": EPOCH + 1,
        "t_ns": clock.ns,
        "time": 1234.5,
        "waited_ns": 3 * SECOND,
        "route_queued_ns": clock.ns,
    }


def test_a_first_registration_has_no_wait_or_route():
    detector, _ = fake()
    detector.registered(STAGE, None, 1, None)
    record = detector.events()[-1]
    assert record["old_epoch"] is None and record["waited_ns"] is None
    assert record["route_queued_ns"] is None


def test_every_reported_time_comes_from_the_nanosecond_clock():
    detector, _ = fake()
    detector.beat(STAGE, EPOCH)
    detector.declare(STAGE, EPOCH, "stream_break")
    detector.registered(STAGE, EPOCH, EPOCH + 1, 5)
    for record in detector.events():
        for name, value in record.items():
            assert "mono" not in name
            if name.endswith("_ns"):
                assert value is None or isinstance(value, int)


def test_events_are_json_bounded_and_logged_as_one_line(caplog):
    detector, _ = fake()
    with caplog.at_level(logging.WARNING, logger="pllm.detection"):
        detector.declare(STAGE, EPOCH, "stream_break", "UNAVAILABLE")
    assert json.loads(caplog.records[0].getMessage()) == detector.events()[0]
    assert "\n" not in caplog.records[0].getMessage()
    for epoch in range(MAX_EVENTS + 10):
        detector.step_deadline("r", epoch)
    assert len(detector.events()) == MAX_EVENTS
