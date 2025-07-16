# Network-based-Treatment-Effect-Estimation-Simulation

# Treatment Effect Estimation with Network Interference

본 프로젝트는 네트워크 상호작용이 존재하는 환경에서의 인과효과 추정 시, 전통적인 추정량들과 네트워크 정보를 활용한 추정량(netAIPW)의 성능을 비교하는 시뮬레이션입니다.  
논문 ["Treatment Effect Estimation with Observational Network Data using Machine Learning"](https://arxiv.org/abs/2201.13268) 를 기반으로 구현되었습니다.

## 🔍 Overview

- **목표**: Spillover effect가 존재하는 상황에서 기존 IPW, Hajek 추정량과 네트워크 정보를 반영한 netAIPW 추정량의 성능 비교
- **환경**: Barabási–Albert 네트워크 구조를 바탕으로 confounder, treatment, outcome을 생성하고, cross-fitting 및 bootstrap을 통해 추정값 및 분산 계산
- **비교 방법**:  
  - netAIPW (network-aware doubly robust estimator)  
  - IPW (Inverse Probability Weighting)  
  - Hajek estimator

## 🧪 Simulation 설정

- BA 네트워크 노드 수 `n`: 625, 1250, 2500
- 연결 수 `m`: 
  - Const 설정: 고정 (예: m=3)
  - Growing 설정: m = 0.0025 * n (예: 1, 3, 6)
- 반복 횟수: 500회
- 추정량 계산: dependency-aware cross-fitting, random forest 사용

## 🛠 실행 방법

### 1. 모듈 기반 단일 시뮬레이션 실행
```bash
python run_BA_simulation.py
