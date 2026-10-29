"""Integer-clock settling and hold gates; every substep is mandatory."""
from __future__ import annotations
from dataclasses import dataclass
import math

import numpy as np


@dataclass
class SettlingGate:
    dt: float
    window: float = .5
    last_step: int = 0
    quiet_start: int | None = None
    passed_step: int | None = None
    ever_overflow: bool = False

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in (self.dt, self.window)):
            raise ValueError('Positive finite dt/window required')

    def observe(self, step, q, v, overflow=0):
        if type(step) is not int or step != self.last_step + 1:
            raise ValueError('Every consecutive physical substep must be observed')
        if q.ndim != 2 or q.shape[1] != 3 or q.shape != v.shape or not len(q):
            raise ValueError('Nonempty matching particle position/velocity required')
        if type(overflow) is not int or overflow < 0:
            raise ValueError('Explicit nonnegative overflow count required')
        self.last_step = step
        self.ever_overflow |= bool(overflow)
        finite = bool(np.isfinite(q).all() and np.isfinite(v).all())
        rms, maximum = None, None
        if finite:
            # Preserve the original float32 arithmetic exactly for ordinary states.
            with np.errstate(over='ignore', invalid='ignore'):
                rms = float(np.sqrt(np.mean(np.sum(v ** 2, axis=1))))
                maximum = float(np.linalg.norm(v, axis=1).max())
            if not math.isfinite(rms) or not math.isfinite(maximum):
                # A scale-first fallback avoids squaring huge finite velocities.
                scale = float(np.abs(v).max())
                unit = v.astype(np.float64) / scale if scale else np.zeros_like(v, dtype=np.float64)
                with np.errstate(over='ignore', invalid='ignore'):
                    rms = float(scale * np.sqrt(np.mean(np.sum(unit * unit, axis=1))))
                    maximum = float(scale * np.linalg.norm(unit, axis=1).max())
                rms = rms if math.isfinite(rms) else None
                maximum = maximum if math.isfinite(maximum) else None
        outside = int(np.count_nonzero((np.abs(q[:, 0] - .65) > .5) | (np.abs(q[:, 1]) > 1.))) if finite else None
        metric_finite = rms is not None and maximum is not None
        low = (finite and metric_finite and rms < .005 and maximum < .05 and outside == 0
               and bool(np.all(q[:, 2] > .798)) and not self.ever_overflow)
        if not low:
            self.quiet_start = None
        elif self.quiet_start is None:
            self.quiet_start = step
        # Integer duration removes floating-point boundary ambiguity.
        quiet_steps = 0 if self.quiet_start is None else step - self.quiet_start
        passed = quiet_steps >= math.ceil(self.window / self.dt - 1e-10)
        if passed and self.passed_step is None:
            self.passed_step = step
        guard = ('nonfinite_state' if not finite else
                 'unrepresentable_speed_metric' if not metric_finite else
                 'extreme_speed_or_position' if maximum > 20. or np.abs(q).max() > 10. else
                 'self_contact_buffer_overflow' if overflow else None)
        return dict(step=step, time_s=step*self.dt, finite=finite,
                    rms_speed_m_s=rms, max_speed_m_s=maximum,
                    vertices_outside_table=outside,
                    min_height_m=float(q[:, 2].min()) if finite else None,
                    overflow=overflow, low_motion=bool(low), quiet_steps=quiet_steps,
                    quiet_duration_s=quiet_steps*self.dt, passed=bool(passed), guard=guard)


@dataclass
class SettlingAndHold:
    """A settling deadline followed by a fully budgeted, non-restarting hold.

    A late first pass does not shorten the hold. Any post-pass quiet failure
    permanently fails this attempt, even if subsequent samples become quiet.
    Wall/guard limits may truncate observation but can never create a pass.
    """
    dt: float
    settling_seconds: float
    hold_seconds: float
    last_step: int = 0
    first_pass: int | None = None
    hold_failed_steps: int = 0

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in
                   (self.dt, self.settling_seconds, self.hold_seconds)):
            raise ValueError('Positive finite timing settings required')
        self.deadline_step = math.ceil(self.settling_seconds / self.dt - 1e-10)
        self.hold_steps = math.ceil(self.hold_seconds / self.dt - 1e-10)
        self.maximum_steps = self.deadline_step + self.hold_steps
        self.stopped = False

    def observe(self, row, wall_limit=False):
        step = row['step']
        if self.stopped or type(step) is not int or step != self.last_step + 1:
            raise ValueError('Every active physical substep must be observed once')
        self.last_step = step
        if self.first_pass is None and row['passed'] and step <= self.deadline_step:
            if not row['low_motion'] or row['guard']:
                raise ValueError('Inconsistent motion pass')
            self.first_pass = step
        elif self.first_pass is not None and not row['low_motion']:
            self.hold_failed_steps += 1
        elapsed = 0 if self.first_pass is None else step - self.first_pass
        complete = self.first_pass is not None and elapsed >= self.hold_steps
        if row['guard']:
            status, stopping = 'failed_guard', True
        elif self.first_pass is None:
            status = 'wall_limit_not_settled' if wall_limit else 'time_limit_not_settled'
            stopping = bool(wall_limit or step >= self.deadline_step)
        else:
            status = ('motion_gate_passed_hold_failed' if self.hold_failed_steps else
                      'motion_gate_and_hold_passed' if complete else
                      'motion_gate_passed_hold_incomplete')
            stopping = bool(wall_limit or complete)
        self.stopped = stopping
        return dict(status=status, stopping=stopping,
                    post_hold_observed_seconds=elapsed*self.dt,
                    post_hold_observation_complete=bool(complete),
                    post_gate_failed_substeps=self.hold_failed_steps,
                    first_motion_pass_time_s=None if self.first_pass is None else self.first_pass*self.dt,
                    wall_limit_reached=bool(wall_limit))
