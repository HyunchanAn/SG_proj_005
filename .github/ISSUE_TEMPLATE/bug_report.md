name: Bug report
description: 시스템 구동 중 발견된 비정상적인 버그나 에러를 보고합니다.
title: "[BUG] "
labels: ["bug"]
body:
  - type: markdown
    attributes:
      value: |
        버그를 신속하게 해결할 수 있도록 아래 항목들을 최대한 상세히 기술해 주세요.
  - type: textarea
    id: description
    attributes:
      label: 버그 설명
      description: 어떤 버그가 발생했는지 명확하고 간결하게 설명해 주세요.
    validations:
      required: true
  - type: textarea
    id: reproduction
    attributes:
      label: 재현 단계
      description: 버그를 다시 유도하기 위한 구체적인 실행 순서를 적어주세요.
      placeholder: |
        1. 'python train.py' 실행
        2. ...
    validations:
      required: true
  - type: textarea
    id: expected
    attributes:
      label: 예상했던 정상 동작
      description: 버그가 발생하지 않았을 때 기대했던 정상적인 결과는 무엇입니까?
    validations:
      required: true
  - type: textarea
    id: logs
    attributes:
      label: 로그 및 스크린샷
      description: 에러 추적 로그(Traceback)나 화면 스크린샷이 있다면 첨부해 주세요.
