from scripts.ops.watch_day_feasibility import jobs_finished, render


def job(stage, success=None):
    r={'status':{'stage':stage}}
    if success is not None:r['result']={'completed':True,'attempts':3,'compliant_successes':success}
    return r


def test_waiting_is_not_finished_or_zero_success():
    s={'direct-bc-v1':job('completed',0),'residual-direct-bc-v1':job('waiting_for_direct_result')}
    assert not jobs_finished(s)
    assert '판정이 보류' in render(s,{})


def test_failed_jobs_are_not_learning_failures():
    s={k:job('failed') for k in ['direct-bc-v1','residual-direct-bc-v1']}
    assert jobs_finished(s)
    assert '판정이 보류' in render(s,{})


def test_success_and_two_valid_failures_have_distinct_decisions():
    s={'direct-bc-v1':job('completed',3),'residual-direct-bc-v1':job('skipped')}
    assert '조건부 진행' in render(s,{})
    s={k:job('completed',0) for k in s}
    assert '대규모 실험 확대는 중지' in render(s,{})
