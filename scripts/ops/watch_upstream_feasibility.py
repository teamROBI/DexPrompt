"""Read-only, bounded SSH watcher for the one official action-backend trial."""
from __future__ import annotations

# Support direct CLI execution and legacy sibling imports after relocation.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[1]))
from _bootstrap import setup as _layout_setup
_layout_setup()


import argparse
import json
from pathlib import Path
import subprocess
import time

REMOTE = '''
import json
from pathlib import Path
p=Path('/data1/yspark/projects/dexprompt/student-checks/upstream-action-v1')
d={}
for name in ['status','result','stages','backend-check']:
 f=p/(name+'.json')
 if f.exists(): d[name]=json.loads(f.read_text())
f=p/'evaluation/summary.json'
if f.exists():
 s=json.loads(f.read_text()); d['replays']=[]
 for e in s.get('episodes',[]):
  for r in e['outcome'].get('replays',[]):
   d['replays'].append({k:r.get(k) for k in ['seed','ticks','failure_reason','contact_acquisition_tick','teacher_compliant_lift','max_height_m']})
f=Path('/data1/yspark/projects/dexprompt/student-runs/upstream-action-v1/progress.json')
if f.exists():d['progress']=json.loads(f.read_text())
print(json.dumps(d))
'''


def verdict(data):
    result = data.get('result', {})
    complete = (data.get('status', {}).get('stage') == 'completed'
                and result.get('completed') is True and result.get('attempts') == 3
                and type(result.get('compliant_successes')) is int
                and 0 <= result['compliant_successes'] <= 3)
    if not complete:
        return '판정 보류', '학습/평가의 정상 완료 자료가 부족하다. 인프라 오류·시간 초과·미실행을 완주0으로 세지 않는다.'
    if result['compliant_successes'] > 0:
        return '제한적 진행 근거 확보', '알려진 학습 과제의 폐루프 완주를 확보했다. 다음은 작은 배치 변화와 목표 변경 평가다. 미지 물체·지시 일반화를 확인한 것은 아니다.'
    return '현재 경로의 확대 보류 유지', '공식 행동 backend 교체만으로 학습한 고정 과제의 완주를 얻지 못했다. 같은 과제의 추가 architecture 튜닝을 중단한다. 논문 아이디어 전체의 불가능을 뜻하지는 않는다.'


def render(data, error=None):
    title, decision = verdict(data)
    result = data.get('result', {})
    score = (f"{result.get('compliant_successes')}/{result.get('attempts')}"
             if title != '판정 보류' else '판정 자료 없음')
    return '\n'.join([
        '# 공식 행동 backend — 3시간 비교 최종 집계', '', time.strftime('%Y-%m-%d %H:%M:%S %Z'), '',
        f'**{title}.** {decision}', '', f'- 최종 접촉 준수 완주: **{score}**.',
        '- 조건: 동일 HX5 데이터/관측/품질, 공식 Transformer·DDIM·optimizer 그룹, 최종epoch13, seed1000–1002.',
        '- 구현 묶음 비교이며 어떤 한 변경이 결과의 원인인지 분리하지 않았다.',
        '- 공식 전체 정책/사전학습 행동 가중치 재현은 아니다. 관측v2/goal-memory/27관절은 기존 HX5 경로다.',
        '- 원격 세부 상태/단계 exit/timeout/평가값은 `snapshot.json`, 계획·코드 차이는 `README.md`.',
        '- 기존 x0·low-noise·direct·residual 최종 평가는 각각0/3. 알려진 과제 반복이며 일반화 성공률이 아니다.',
        '- Claude의 손 모양 편집 결과는 별도 확인 전 이 집계에 포함하지 않는다.',
        '- 이 감시자는 추가 학습·평가·체크포인트 선택을 실행하지 않는다.',
        *(['', '마지막 조회 오류: '+error] if error else []), '',
    ])


def write(path, data):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(data if isinstance(data, str) else json.dumps(data, indent=2)+'\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'FINAL.md').exists():
        raise FileExistsError('final report already exists')
    data = {}
    while True:
        error = None
        try:
            proc = subprocess.run(
                ['ssh', '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'ConnectTimeout=10',
                 '-i', '/home/yspark/.ssh/id_rsa_yspark', 'lecun', 'python3', '-'],
                input=REMOTE, text=True, capture_output=True, timeout=35, check=True,
            )
            data = json.loads(proc.stdout)
            write(args.out/'snapshot.json', {'queried_at':time.time(), **data})
        except (subprocess.SubprocessError, ValueError) as exc:
            error = str(exc)
        expired = time.time() >= args.deadline
        done = not error and data.get('status', {}).get('stage') in ('completed', 'failed', 'blocked')
        write(args.out/'watcher.json', {'running':not(done or expired), 'updated_at':time.time(),
                                      'remote_error':error, 'deadline_reached':expired})
        if done or expired:
            write(args.out/'FINAL.md', render(data, error))
            return
        time.sleep(min(60, max(0, args.deadline-time.time())))


if __name__ == '__main__':
    main()
