from scripts.ops.watch_upstream_feasibility import verdict


def test_infrastructure_failure_is_not_a_zero_success_result():
    assert verdict({'status': {'stage': 'failed'}})[0] == '판정 보류'


def test_completed_zero_and_success_have_distinct_bounded_decisions():
    d = {'status': {'stage': 'completed'},
         'result': {'completed': True, 'attempts': 3, 'compliant_successes': 0}}
    assert verdict(d)[0] == '현재 경로의 확대 보류 유지'
    d['result']['compliant_successes'] = 1
    assert verdict(d)[0] == '제한적 진행 근거 확보'


def test_partial_evaluation_cannot_pass_as_the_fixed_final_evaluation():
    d = {'status': {'stage': 'completed'},
         'result': {'completed': True, 'attempts': 1, 'compliant_successes': 0}}
    assert verdict(d)[0] == '판정 보류'
