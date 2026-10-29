"""Bounded CPU journal writer for explicitly qualified collection runtimes.

Single producer; immutable host copies; ordered writes per journal. The caller
must flush before verifying/publishing any completed trajectory. This class does
not change simulation, journal serialization, fsync or integrity semantics.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading
from types import FunctionType, SimpleNamespace

CONTRACT = dict(workers=4, max_bytes=64*1024**2, max_pending=32)


def simulate(grouped, *args, **kwargs):
    """Run unchanged simulator code with private journal bindings, never globals.

    The frozen launch bundle binds this adapter as well as the original code.
    Verification drains the queue before any per-attempt completion receipt.
    No process-global monkeypatch is needed; serial qualification stays original.
    """
    original = grouped.simulate
    with BoundedFrameWriter(**CONTRACT) as writer:
        def journal(*a, **kw):
            actual = grouped.FrameJournal(*a, **kw)
            return SimpleNamespace(root=actual.root,
                append=lambda step, arrays: writer.append(actual, step, arrays))
        def verify(*a, **kw):
            writer.flush()
            return grouped.verify_journal(*a, **kw)
        namespace = dict(original.__globals__, FrameJournal=journal, verify_journal=verify)
        run = FunctionType(original.__code__, namespace, original.__name__,
                           original.__defaults__, original.__closure__)
        run.__kwdefaults__ = original.__kwdefaults__
        result = run(*args, **kwargs)
        writer.flush()
        return result


class BoundedFrameWriter:
    def __init__(self, *, max_bytes, workers=2, max_pending=16):
        if type(max_bytes) is not int or max_bytes<1:raise ValueError('Positive byte limit required')
        if type(workers) is not int or not 1<=workers<=8:raise ValueError('workers must be 1..8')
        if type(max_pending) is not int or not 1<=max_pending<=64:raise ValueError('max_pending must be 1..64')
        self.max_bytes=max_bytes;self.max_pending=max_pending
        self.pending_bytes=0;self.peak_pending_bytes=0
        self._owner=threading.get_ident();self._closed=False;self._failure=None
        self._pending=deque();self._last={};self._counts={}
        self._pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='mpm-frame-write')

    def _check(self):
        if threading.get_ident()!=self._owner:raise RuntimeError('Single producer thread required')
        if self._closed:raise RuntimeError('Writer is closed')
        if self._failure is not None:raise self._failure

    def _resolve(self):
        future,size=self._pending.popleft()
        try:future.result()
        except BaseException as exc:
            if self._failure is None:self._failure=exc
            raise
        finally:self.pending_bytes-=size

    def append(self,journal,step,arrays):
        import numpy as np
        from collection.mpm.journal import FIELDS
        self._check()
        while self._pending and self._pending[0][0].done():self._resolve()
        count=self._counts.get(journal,journal.count)
        if (set(arrays)!=FIELDS or step!=count*journal.clock.substeps_per_transition
                or count>journal.clock.transitions):raise ValueError('Journal fields or submitted clock mismatch')
        if any(not isinstance(a,np.ndarray) or a.dtype.kind not in 'biufc' for a in arrays.values()):
            raise ValueError('Numeric host arrays required')
        size=sum(a.nbytes for a in arrays.values())
        if size>self.max_bytes:raise ValueError('One frame exceeds the byte budget')
        while self._pending and (self.pending_bytes+size>self.max_bytes or len(self._pending)>=self.max_pending):
            self._resolve()
        # Check capacity before copying. The limit covers owned queued/active copies,
        # not caller buffers, compression scratch space, or full-trajectory arrays.
        copied={k:np.array(a,copy=True) for k,a in arrays.items()}
        for a in copied.values():a.flags.writeable=False
        previous=self._last.get(journal)
        def write():
            if previous is not None:previous.result()
            journal.append(step,copied)
        future=self._pool.submit(write)
        self._last[journal]=future;self._counts[journal]=count+1
        self._pending.append((future,size));self.pending_bytes+=size
        self.peak_pending_bytes=max(self.peak_pending_bytes,self.pending_bytes)

    def flush(self):
        self._check()
        while self._pending:self._resolve()

    def close(self):
        if threading.get_ident()!=self._owner:raise RuntimeError('Single producer thread required')
        if self._closed:
            if self._failure is not None:raise self._failure
            return
        try:
            while self._pending:
                try:self._resolve()
                except BaseException:pass  # Drain owned work; report first error below.
        finally:
            self._pool.shutdown(wait=True);self._closed=True
        if self._failure is not None:raise self._failure

    def __enter__(self):self._check();return self

    def __exit__(self,kind,value,traceback):
        try:self.close()
        except BaseException:
            if kind is None:raise
        return False
