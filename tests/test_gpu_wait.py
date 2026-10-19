"""_fit_waiting_for_gpu: a block that runs out of GPU memory waits for memory and fits again.

The GPU is faked: these tests need no GPU."""
import pytest
import torch

import segmentNMF.distributed_NMF_methods as dm

GIB = 2**30


def oom(gib=2.0):
    return torch.OutOfMemoryError(
        'CUDA out of memory. Tried to allocate {:.2f} GiB. GPU 0 has a total capacity '
        'of 80.00 GiB of which 1.00 GiB is free.'.format(gib))


@pytest.fixture
def gpu(monkeypatch):
    """A fake card: free memory from a list, one value per mem_get_info call."""
    state = {'free': [], 'total': 80 * GIB, 'peak': 0, 'emptied': 0, 'slept': []}

    def mem_get_info():
        free = state['free'].pop(0) if len(state['free']) > 1 else state['free'][0]
        return free, state['total']

    def empty_cache():
        state['emptied'] += 1

    monkeypatch.setattr(torch.cuda, 'mem_get_info', mem_get_info)
    monkeypatch.setattr(torch.cuda, 'empty_cache', empty_cache)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: state['peak'])
    state['sleep'] = lambda s: state['slept'].append(s)
    return state


def fit_failing(times, error=oom):
    calls = []

    def fit():
        calls.append(1)
        if len(calls) <= times:
            raise error()
        return 'result'
    return fit, calls


def test_requested_bytes_reads_the_failed_allocation():
    assert dm._requested_bytes(oom(3.33)) == int(3.33 * GIB)
    assert dm._requested_bytes(RuntimeError('Tried to allocate 20.00 MiB.')) == 20 * 2**20
    assert dm._requested_bytes(RuntimeError('no size here')) == 0


def test_a_fit_without_error_runs_once_and_releases_the_cache(gpu):
    fit, calls = fit_failing(0)
    assert dm._fit_waiting_for_gpu(fit, sleep=gpu['sleep']) == 'result'
    assert len(calls) == 1
    assert gpu['emptied'] == 1
    assert gpu['slept'] == []


def test_a_busy_card_makes_the_block_wait_then_fit_again(gpu):
    gpu['peak'] = 10 * GIB                      # the failed try needs 10 + 2 GiB
    gpu['free'] = [5 * GIB, 5 * GIB, 11 * GIB, 12 * GIB]   # after the error, then per check
    fit, calls = fit_failing(1)
    assert dm._fit_waiting_for_gpu(fit, poll_s=30, sleep=gpu['sleep']) == 'result'
    assert len(calls) == 2
    assert gpu['slept'] == [30, 30, 30]         # 5, 11 GiB are too little; 12 GiB is enough
    assert gpu['emptied'] == 2                  # after the failed try and after the good one


def test_a_block_larger_than_the_card_fails_at_once(gpu):
    gpu['peak'] = 79 * GIB
    gpu['free'] = [80 * GIB]
    fit, calls = fit_failing(5)
    with pytest.raises(torch.OutOfMemoryError, match='more than the 80.0 GiB of the whole GPU'):
        dm._fit_waiting_for_gpu(fit, sleep=gpu['sleep'])
    assert len(calls) == 1
    assert gpu['slept'] == []


def test_the_wait_has_a_limit(gpu):
    gpu['peak'] = 10 * GIB
    gpu['free'] = [1 * GIB]
    fit, calls = fit_failing(5)
    with pytest.raises(torch.OutOfMemoryError, match='waited 90 s for 12.0 GiB'):
        dm._fit_waiting_for_gpu(fit, poll_s=30, max_wait_s=90, sleep=gpu['sleep'])
    assert len(calls) == 1
    assert gpu['slept'] == [30, 30, 30]


def test_the_limit_counts_all_waits_of_the_block(gpu):
    gpu['peak'] = 10 * GIB
    gpu['free'] = [20 * GIB]                    # enough memory at every check
    fit, calls = fit_failing(5)
    with pytest.raises(torch.OutOfMemoryError, match='waited 60 s'):
        dm._fit_waiting_for_gpu(fit, poll_s=30, max_wait_s=60, sleep=gpu['sleep'])
    assert len(calls) == 3                      # 2 waits of 30 s, so 3 tries
    assert gpu['emptied'] == 3


def test_other_errors_are_not_retried(gpu):
    fit, calls = fit_failing(1, error=lambda: ValueError('bad input'))
    with pytest.raises(ValueError, match='bad input'):
        dm._fit_waiting_for_gpu(fit, sleep=gpu['sleep'])
    assert len(calls) == 1
    assert gpu['emptied'] == 1
    assert gpu['slept'] == []
