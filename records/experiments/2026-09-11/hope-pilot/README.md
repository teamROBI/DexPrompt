<!-- research-scope-2026-09-12 -->
> **현재 연구와의 관계:** [두 단계 정의](../../../../RESEARCH_PLAN.md)를 기준으로 해석한다.
> 아래는 해당 날짜·설정의 실험 기록이며 수치와 판정은 보존했다. “다음/미실행/진행 중”은 당시 상태다.
> 생성·접근·시각 BC의 개별 결과를 형상 기반 선정/적응 시스템이나 최종 일반화의 증거로 확대하지 않는다.

# HOPE 파일럿 자산·텍스처·30개 초기 배치 검증

2026-09-11. 준비 대상 6개 중 **3개 물체 × 5개 배치 = 15개 초기 장면**이
사전 검사를 통과했다. **파지 성공이나 학습 시범 15개를 뜻하지 않는다.**
teacher 생성·정련·실행·추가 학습은 하지 않았다.

## 구현과 생성한 자산

- `scripts/prepare_hope_pilot.py`: 공식 full OBJ의 감사 당시 해시를 확인하고
  0.001 변환·X축 90° 회전·바닥 정렬을 적용한다. 기존 Mustard/ChocolatePudding은
  별도로 복사하고, AlphabetSoup/Peaches/BBQSauce/Raisins는 기본 CoACD(seed 0)로
  생성했다. 한 번에 하나씩, CPU/BLAS 1스레드·nice 15·물체당 300초 상한이다.
  종료 코드·시간 초과를 기록하며 실패한 디렉터리는 usable objects로 옮기지 않는다.
  **6/6 생성 완료**, timeout 0. 가장 긴 AlphabetSoup은 약 275초였다.
- `src/icra/object_visuals.py`와 `scene.build`: 별도 장면 설정의
  `object_appearance: texture_v1`에서만 OBJ/MTL/UV/이미지를 연결한다. 누락된 재료는
  회색으로 대체하지 않고 실패한다. MuJoCo 3.3.3이 JPEG 파일을 직접 읽지 못하는 것을
  실제 자산 컴파일에서 확인해, PNG를 메모리 VFS에 만들어 전달한다. 원본 이미지는
  변경하지 않고 MJB에 픽셀이 포함된다. 기존 설정의 기본값은 solid다.
- `configs/scene_hope_textured.yaml`: 새 텍스처 장면. 기존 256×256/머리 카메라
  아래로 20°인 visual profile을 그대로 적용한다. 사진은 [objects.png](objects.png).
- 장면의 `object_yaw_degrees`를 공통 builder에 연결했다. 관측·필터·정렬·실행이
  같은 base yaw를 사용한다. **해당 장면으로 실행할 때 에피소드 yaw는 0**이어야 한다.
  Recorder의 episode yaw는 base yaw에 추가되는 회전이다. 이미 15°인 장면에 다시
  episode yaw 15°를 주면 30°가 되므로 `generation-plan.json`에도 구분해 기록했다.

자산: `out/hope-pilot-v1/objects/`. 단위·치수·출처는
[전수 자산 감사](../../2026-09-10/hope-data-readiness/README.md)와 연결된다.

## 30개 장면의 실측

각 물체에 중앙, yaw −15/+15°, X −20/+20 mm의 다섯 조건을 사용했다.
전체 조합이 아니라 한 요인씩 바꾼 다섯 조건이다. 각 조건별 scene.yaml을 저장하고
실제 Recorder와 관측/필터/정렬이 사용하는 `scene.build`의 초기 world transform을
대조했다. **30/30 위치·회전 일치**. 미터 스케일은 1.0(메시에 이미 변환 적용)이다.

실제 Recorder의 초기화·brace 후 기본 settle 600 physics steps(1.2초)를 사용했다.
시각 입력은 native 256×256 두 카메라에서 RGB와 segmentation으로 확인했다.

| 물체 | 충돌 조각 | 컴파일 질량 (kg) | 초기 장면 통과 | 보류 사유 |
| --- | ---: | ---: | ---: | --- |
| AlphabetSoup | 63 | 0.10629 | **5/5** | — |
| Peaches | 45 | 0.10779 | **5/5** | — |
| Mustard | 2 | 0.10933 | 0/5 | 1.2초 후 각속도 약 0.190 rad/s |
| BBQSauce | 2 | 0.10940 | 0/5 | 1.2초 후 각속도 약 0.133 rad/s |
| ChocolatePudding | 1 | 0.10704 | 0/5 | 1.2초 후 각속도 약 0.355 rad/s |
| Raisins | 7 | 0.10814 | **5/5** | — |

**이번 초기 장면 준비 기준**은 놓임 이동 ≤5 mm, 회전 ≤5°,
마지막 선속도 ≤0.005 m/s·각속도 ≤0.05 rad/s, 로봇–물체/테이블 침투 ≤1 µm,
그리고 카메라 하나 이상에서 기존 goal-reference 품질 기준 통과다.
이 기준은 새 자산을 거르는 preflight이며 **기존 파지 성공 기준을 수정한 것이 아니다.**
손가락의 초기 자세는 기본 Recorder 자세이며, 선정할 파지 유형의 calibrated open에서의
전체 접근 및 가시성은 이후 파지 실행에서 다시 검증해야 한다.

보류한 세 물체는 중앙 배치에서 **같은 명령·물리 설정으로 4초까지** 추가 관찰했다.
1.2초 시점의 자세가 원래 검사와 일치함을 확인했고, 마지막 0.5초 구간의 속도 검사도
셋 모두 실패했다. 최종 각속도는 Mustard 0.414, BBQSauce 0.140,
ChocolatePudding 0.260 rad/s였다. 따라서 단순히 1.2→4초로 대기 시간을 늘려
해결됐다고 할 수 없다. 마찰·질량·성공 기준을 조정하거나 여섯 물체를 전부
통과한 것으로 만들지 않았다. 원인은 아직 분리하지 않았으며 이 세 물체의 추가
튜닝을 다음 수집의 선행 조건으로 두지 않는다.

## 물리와 teacher 계약 검증

- 각 물체 중앙 조건에서 solid와 texture 모델을 같은 integration state에서
  600 physics steps 진행했다. **6/6 최종 integration state가 배열까지 동일**하다.
- 실제 Dexonomy `_add_rigid_object`와 `op=eval`의 obj_mass=0.1 밀도 규칙으로
  물체를 컴파일했다. 실행기의 질량·관성과 **6/6 일치**:
  최대 질량 차이 1.39e-17 kg, 대각 관성 차이 5.15e-19 kg·m².
  이는 물체 로더/질량 계약 검사이며 ADMM 전체 실행을 검증한 것은 아니다.
- 초기 물체 좌표계 테이블 z=0과 실행기 테이블 배치를 확인했다. 안정화 후 실제
  물체 변환도 별도로 기록했다. 향후 예측·정련 단계에서 이 이동량을 무시해도 된다고
  가정하지 않는다.
- 새 자산의 scene config에는 table 항목이 있다. 정련에는 기존
  `scripts/dexonomy_table_scene` 훅과 `ICRA_DEXONOMY_TABLE_SCENE=1`이 필요하다.
  훅 없이 기본 단일 항목 assertion에서 실패한 결과를 파지 수율 0으로 세면 안 된다.
- 합성 실제 렌더 검사에서 RGB 변화, segmentation 보존, 물리 상태 보존, MJB의
  texture/UV 보존을 확인했다. 실물 6개도 inspection 렌더를 만들고 확인했다.

## 다음 수집 대상으로 고정한 것

`out/hope-pilot-v1/ready-objects/`는 **AlphabetSoup·Peaches·Raisins만** 가리킨다.
[generation-plan.json](generation-plan.json)에 15개 초기 장면의 YAML/메시 해시,
world transform, 보류 15개, template 파일 해시를 기록했다.

| 물체 | 사전 선택 유형(형상 기반 가설, 아직 미검증) |
| --- | --- |
| AlphabetSoup / Peaches | `10_Power_Disk`, `1_Large_Diameter` |
| Raisins | `9_Palmar_Pinch`, `1_Large_Diameter` |

다음은 **세 물체의 중앙 조건부터** GraspGenX→align→refine→접근·닫기·리프트·hold다.
align seed 0/1/2, 정련 400회, 물체당 동적 실행 최대 12회로 고정한다.
초기 중앙 조건 총 상한은 **36회**다. 접촉군을 줄이거나 실패 물체의 유형을 계속
추가하지 않는다. 중앙 조건에서 완주한 물체만 다른 네 배치로 확대한다.
고유 물체/배치/teacher 수와 시간당 채택 시범 수로 수율을 판단한다.

현재 저장한 그림은 초기 장면 점검 자료다. action 시범이 없으므로 VLA 학습
데이터셋으로 섞지 않았다. 이번 15개에서 성공 파지가 나오는지는 다음 실험의 질문이다.

## 산출물과 검사

- [build.json](build.json): 생성 단계 상태·종료 코드·시간·해시.
- [checks.json](checks.json): 30개 초기 장면의 배치·시야·안정화 결과.
- [teacher-contract.json](teacher-contract.json): teacher 로더/실행기 질량·관성 대조.
- [settling-tail.json](settling-tail.json): 보류한 세 물체의 4초 관찰.
- `out/hope-pilot-v1/checks-v3/<object>/<placement>/`: YAML, head/wrist RGB, head mask.
  v1은 미지원 Recorder CLI 인자, v2는 JPEG 직접 입력 오류에서 멈춘 개발 시도다.
  둘은 완료 결과나 분모에 포함하지 않는다. v3이 완성된 30조건 검사다.
- 관련 테스트 **72개 통과**(신규 12개 포함), 변경 코드 Ruff 및 `git diff --check` 통과.
  전체 저장소 테스트를 실행한 것은 아니다. 새 학습·teacher 파지 실험 없음.
