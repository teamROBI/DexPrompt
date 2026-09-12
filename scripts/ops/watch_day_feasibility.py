"""Read-only experiment watcher; write a bounded, conservative final decision.

No training, model selection, or changes to the teacher/student jobs. Designed to
run in tmux so the user's network and the interactive Codex turn are not required.
"""
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

TERMINAL = {'completed', 'failed', 'blocked', 'skipped'}
REMOTE = r'''
import json
from pathlib import Path
b=Path('/data1/yspark/projects/dexprompt/student-checks')
out={}
for name in ['direct-bc-v1','residual-direct-bc-v1']:
 r=b/name; row={'source':str(r)}
 for file,key in [('status.json','status'),('result.json','result'),('stages.json','stages')]:
  p=r/file
  if p.exists():
   d=json.loads(p.read_text())
   if key=='stages':d=[{k:x.get(k) for k in ['stage','pid','exit_code','timed_out','started_at','finished_at']} for x in d]
   row[key]=d
 p=r/'evaluation/summary.json'
 if p.exists():
  d=json.loads(p.read_text()); row['replays']=[]
  for e in d.get('episodes',[]):
   for x in e['outcome'].get('replays',[]):
    row['replays'].append({k:x.get(k) for k in ['seed','ticks','failure_reason','contact_acquisition_tick','teacher_compliant_lift','max_height_m']})
 out[name]=row
p=b/'proprio-retrieval-v1/result.json'
if p.exists():out['retrieval']=json.loads(p.read_text())['outcome']
print(json.dumps(out))
'''


def write(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(value if isinstance(value,str) else json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def jobs_finished(snapshot):
    return all(snapshot.get(name,{}).get('status',{}).get('stage') in TERMINAL
               for name in ['direct-bc-v1','residual-direct-bc-v1'])


def render(snapshot, teacher, *, deadline_reached=False, connection_error=None):
    valid=[]
    rows=[]
    for name in ['direct-bc-v1','residual-direct-bc-v1']:
        job=snapshot.get(name,{})
        result=job.get('result',{})
        stage=job.get('status',{}).get('stage','미확인')
        ok=(stage=='completed' and result.get('completed') is True
            and isinstance(result.get('attempts'),int) and result['attempts']>0
            and isinstance(result.get('compliant_successes'),int))
        if ok:
            valid.append(result['compliant_successes'])
            score=f"{result['compliant_successes']}/{result['attempts']}"
        else:
            score='미실행' if stage=='skipped' else '판정 자료 없음'
        rows.append(f'| {name} | {stage} | {score} |')
    improvements=teacher.get('first_improvement')
    edits_complete=(bool(teacher.get('finished')) and teacher.get('by_role',{}).get('edit')==24)
    if any(x>0 for x in valid):
        decision='조건부 진행: 알려진 한 과제에서 행동 전이의 최소 증거를 확보했다.'
        next_work='다음은 여러 물체·배치의 분리 평가와 동일 장면 목표 전환이다. 한 과제 성공을 미지 물체/지시 일반화로 확대하지 않는다.'
    elif len(valid)==2:
        decision='현재 student 구현의 대규모 실험 확대는 중지하는 편이 합리적이다. 두 제한된 변경으로도 학습한 과제의 완주를 확보하지 못했다.'
        next_work='같은 설정으로 데이터·학습 시간만 늘리기보다, 검증된 행동 학습 구현의 적용 또는 복귀 시범을 포함한 작은 전이 검증으로 경로를 바꾼다. 두 단계 아이디어 자체가 불가능하다는 결론은 아니다.'
    else:
        decision='실험 완료 자료가 부족해 학습 가능성 판정이 보류됐다. 인프라 실패·미실행을 파지 실패로 세지 않는다.'
        next_work='누락/실패한 실행 단계만 확인한다. 완료되지 않은 결과로 성공이나 불가능을 선언하지 않는다.'
    lines=['# 당일 가능성 판정 — 자동 집계', '', time.strftime('%Y-%m-%d %H:%M:%S %Z'), '',
           '**'+decision+'**', '',
           '| 시각 BC | 실행 상태 | 접촉 준수 완주 |','| --- | --- | --- |',*rows,'',
           '기준선 x0와 low-noise x0의 사전 고정 최종 평가도 각각0/3이었다. 직접/잔차 정책은 결정론적이며 같은 초기 상태의3회는 독립 일반화 표본이 아니다.', '',
           '## 1단계: Dexonomy 시범 편집', '']
    if edits_complete:
        lines.append(f"Claude 원시 요약: zero-edit6회, 고정 손목 편집24회, 개선된 물체 조건{len(improvements or {})}개. 전체 물리 실행{teacher.get('charged')}회.")
        lines.append('손 모양 편집·학습된 유형 선정·편집 RL은 이 비교에 포함되지 않았다. 작은 손목 편집의 무개선은 그 전체 방법의 반증이 아니다.')
    else:
        lines.append('완료된 teacher 편집 요약을 확인하지 못했다. 실패0이나 성공으로 대체하지 않는다.')
    retrieval=snapshot.get('retrieval',{})
    if retrieval.get('teacher_compliant_lift'):
        lines += ['', '비시각 최근접 시범 재사용은 같은 학습 궤적에서 완주했다. 시계·object pose·teacher phase 없이 관절 상태로 검색했지만, 학습 궤적 밖 복귀나 시각 활용을 증명하지는 않는다.']
    lines += ['', '## 논문에 대한 결론', '',
              'DemoFunGrasp와 DexGraspVLA는 두 단계 접근의 방법론적 근거다. 다만 현재 자료만으로 원래 목표 전체가 ICRA/IROS 수준에서 검증됐다고 할 수는 없다.',
              '아직 필요한 핵심은 학습된 선정/적응의 이득, 다양한 형상에서의 자동 수집, 목표 조건 행동 전이, 미지 형상·배치 평가, Dexonomy 기여의 비교다. 이 자동 집계는 논문 채택 확률을 산출하지 않는다.', '',
              next_work, '',
              'HOPE A12는12시범/3물체/5후보의 준비 자료이며, Milk B와 신규 teacher 결과는 자동 병합하지 않았다.', '',
              '## 실행 근거', '',
              '- `snapshot.json`: lecun의 고정 작업 경로에서 읽은 상태/최종 결과/단계 exit/timeout.',
              '- `teacher-summary.json`: Claude의 원시 완료 요약 사본.',
              '- 기존 결과: `../lecun-final-check/README.md`.',
              '- 코드·예산·범위: `../direct-bc-feasibility/README.md`.',
              '- 새 학습·재평가·체크포인트 선택을 이 감시자가 실행하지 않는다.']
    if deadline_reached:
        lines += ['', '**종료 상한에 도달해 집계했다. 미완료 항목은 표에서 구분한다.**']
    if connection_error:
        lines += ['', '마지막 원격 조회 오류: '+connection_error+' (snapshot의 조회 시각을 확인할 것).']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--teacher-summary',type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    snapshot={}
    while True:
        error=None
        try:
            proc=subprocess.run(['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes',
                                 '-o','ConnectTimeout=10','-i','/home/yspark/.ssh/id_rsa_yspark',
                                 'lecun','python3','-'],input=REMOTE,text=True,
                                capture_output=True,timeout=35,check=True)
            snapshot=json.loads(proc.stdout)
            write(a.out/'snapshot.json',{'queried_at':time.time(),'jobs':snapshot})
        except (subprocess.SubprocessError,ValueError) as e:
            error=str(e)
        teacher=json.loads(a.teacher_summary.read_text()) if a.teacher_summary.exists() else {}
        write(a.out/'teacher-summary.json',teacher)
        expired=time.time()>=a.deadline
        finished=not error and jobs_finished(snapshot) and bool(teacher.get('finished'))
        write(a.out/'status.json',{'updated_at':time.time(),'running':not(finished or expired),
                                  'remote_error':error,'deadline_reached':expired})
        if finished or expired:
            write(a.out/'FINAL.md',render(snapshot,teacher,deadline_reached=expired,connection_error=error))
            print('Final report:',a.out/'FINAL.md',flush=True)
            return
        time.sleep(min(60,max(0,a.deadline-time.time())))


if __name__=='__main__':
    main()
