# VQA AQ 25 (🚧 Under Development 🚧)

이 프로젝트는 **Visual Question Answering (VQA)** 시스템에서 질문이 모호할 경우, **추가적인 질문(QG; Question Generation)을 통해 명확한 정보를 획득한 후 다시 VQA를 수행하는 방식**을 적용한다.

🚀 **현재 본 프로젝트는 개발 중(Under Development)이며, 성능 개선 및 최적화를 지속적으로 진행하고 있습니다.**  
📌 **추후 업데이트될 사항**: 질문 생성(QG) 최적화, 실행 시간 단축, 추가적인 평가 METHOD 개선  

## **1. 개요 (Overview)**
본 연구는 모호한 질문에 대해 `"I'm not sure"` 와 같은 특정 출력을 내도록 시스템을 설계하고, 이를 바탕으로 **추가적인 서브 QA (clarification question)** 를 진행한 후 최종적으로 VQA를 수행하는 방법을 개발하였다.

## **2. 진행 방식**
1. **질문의 명확성 확인**
   - VQA 모델이 질문을 받고, 확실하게 대답할 수 있는지 확인  
   - `"Yes"` 인 경우: 바로 VQA 수행  
   - `"I'm not sure"` 인 경우: 추가적인 질문 생성(QG) 후 다시 VQA 진행  

2. **Clarification Question Generation (QG) 진행**
   - 최대 3회 반복하여 질문 명확성 개선  
   - 이후, 개선된 질문을 기반으로 VQA 수행  

## **3. 성능 비교** *(RTX 3090 24GB 기준)*

| 방법 | 수행 시간 | 성능 (정확도) |
|------|----------|--------------|
| **Baseline (기본 VQA)** | 20분 | 197/300 |
| **기존 방법** | 1시간 44분 | 209/300 |
| **개선된 방법 (Clarification 적용)** | 1시간 4분 5초 | 218/300 |
