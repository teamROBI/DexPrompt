# 시작 상태의 범위 — 짧은 확인

2026-09-12. lecun의 승인된 `student-data/hope-twelve-v1`에서 후보별 첫 측정 관절을 읽었다.
첫 Peaches 시범에 대한 손20관절 최대 절대 차이:

| 물체/후보 | 차이(rad) |
| --- | ---: |
| Peaches / 203b95f1 | 0 |
| AlphabetSoup / 057c04e1 | 0 |
| AlphabetSoup / 1443cca6 | 0 |
| Raisins / 0ddf3447 | 0.280417 |
| Raisins / 3e8fa799 | 0.280417 |

`src/icra/grasp_env.py:calibrated_open`은 유형별 gripper config의 open을 읽고,
`src/icra/execution.py:TeacherExecutor.initialize`는 이를 초기 손 관절에 설정한다.
따라서 초기 손 자세가 유형 정보의 단서가 될 수 있다. 이것만으로 정책이 실제로 그 단서를
사용했다거나 시범이 잘못됐다고 결론내리지는 않는다.

현재 검증은 보관된 시작 상태에서의 실행이다. 공통 중립 home에서 목표만 보고 유형을 선택하는
과제를 주장하려면 중립 자세→유형별 pre-shape도 정책 행동/시범에 포함하거나,
외부에서 유형/시작 자세를 제공하는 조건임을 명시해야 한다. 오늘 기존 모델/시범/reset은 바꾸지 않았다.
