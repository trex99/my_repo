# 2‑Wasserstein 거리 추정을 위한 Back‑and‑Forth 알고리즘과 ICNN 메모리 뱅크 확장

## 1. 서론

### 1.1 배경과 문제의식
최적수송은 확률분포 사이의 차이를 기하적으로 포착하는 방법이다. 비용 함수를 \(c(x,y)=\|x-y\|^2\)로 두는 2‑Wasserstein 제곱거리 \(W_2^2\)는 Brenier 사상과 \(c\)-볼록 포텐셜이라는 볼록해석적 구조를 갖지만, 신경망으로 근사할 때 세미듀얼 최적화가 수치적으로 까다롭다. 어려움의 핵심은 두 가지다. 첫째, 미니배치 상황에서 \(c\)-볼록성을 일관되게 강제해야 한다. 둘째, \(c\)-transform과 \(cc\)-transform을 정확하고 안정적으로 계산해야 한다.

### 1.2 연구 목표
본 연구는 생성 모델을 학습하지 않고 \(W_2^2(\mu,\nu)\) 자체를 직접 근사하는 방법을 제시한다. 이를 위해 증명 가능한 하한과 일관성 진단을 동시에 제공하는 신경망 포텐셜을 설계하고, 작은 미니배치에서도 안정적으로 작동하는 학습 절차를 제안한다.

### 1.3 공헌 요약
본 논문은 2-Wasserstein 거리 추정을 위한 새로운 신경망 학습 프레임워크를 제안한다. **핵심 아이디어는 c-볼록성 일관성 갭을 정량화하고, 이를 기반으로 최적화 목표를 동적으로 전환하는 것이다.** 구체적 기여는 다음 세 가지로 요약된다.

**첫째**, 세 가지 목적함수 체계 \(J_1, J_2, J_3\)와 일관성 갭 \(\Delta_\mu, \Delta_\nu\)를 정식화하고, 이들 사이의 부등식과 동치성을 엄밀히 증명한다.

**둘째**, 일관성 갭의 크기에 따라 갭 감소 단계와 하한 상승 단계를 교대로 수행하는 Back‑and‑Forth 학습 알고리즘을 제안한다. 이 전환 메커니즘은 미니배치 환경에서 c-볼록성 보존과 거리 추정 정확도를 균형있게 달성하는 새로운 적응적 학습 전략이다. 학습 단계의 안전성과 단조성 근거를 제시한다.

**셋째**, max‑affine, log‑sum‑exp, 이차 프로토타입 슬롯으로 구성된 메모리‑뱅크 레이어를 포함하는 입력‑볼록 신경망을 제안한다. 이 구조가 입력 볼록성을 보존하면서도 지역 표현력을 확장하여 \(c\)-transform 근사 오차를 줄임을 이론적으로 설명한다.

본 방법의 하이퍼파라미터는 명확한 수학적 의미를 갖는다. 핵심 파라미터인 갭 임계값 \(\tau\)는 c-볼록성 허용 오차를 직접 제어하며, WGAN-GP의 gradient penalty처럼 손실 항 간 균형을 맞추는 휴리스틱 가중치가 아니다. 실험에서 \(\tau\)를 데이터셋 간 크기 스케일에 비례하여 설정하면 추가 튜닝 없이 일관된 성능을 얻을 수 있었다 (섹션 9 참조).

본 연구는 분포 간 2‑Wasserstein 거리의 신경망 기반 근사 문제에 집중하며, 생성 모델 학습이나 샘플 생성과 같은 주제는 다루지 않는다.

---

## 2. 관련 연구
세 가지 축을 중심으로 문헌을 개관한다. 첫째, \(W_2\)에 대한 Kantorovich 이중성과 세미듀얼. 둘째, ICNN과 Brenier 포텐셜을 이용한 신경망 기반 최적수송. 셋째, 본문의 목적함수와 부등식을 떠받치는 볼록해석 도구인 Fenchel–Young 부등식과 공액함수 이론. 참고문헌의 상세 목록은 뒷부분에서 제시한다.

---

## 3. 표기와 기본 가정
- 배경공간은 \(\mathbb{R}^d\)이고 노름은 유클리드 노름 \(\|\cdot\|\)이다. 비용은 \(c(x,y)=\|x-y\|^2\)로 둔다.
- \(\mathbb{R}^d\) 위의 확률측도 \(\mu,\nu\)는 2차 모멘트가 유한하다. 기댓값은 \(\mathbb{E}_\mu[f]=\int f\,d\mu\)로 표기한다.
- 함수 \(f\)의 \(c\)-transform과 \(cc\)-transform은 다음과 같다.
  
  \[
  f^{\,c}(y)=\inf_{x\in\mathbb{R}^d}\{\|x-y\|^2-f(x)\},\qquad f^{cc}(x)=\inf_{y\in\mathbb{R}^d}\{\|x-y\|^2-f^{\,c}(y)\}.
  \]
- 함수 \(\varphi\)가 어떤 \(\psi\)에 대해 \(\varphi=\psi^{\,c}\)이면 \(c\)-볼록이라고 한다.
- **Brenier 파라미터화:** Brenier 정리 (Brenier, 1991; Villani, 2009)에 의해 최적 수송 맵은 \(T(x) = \nabla u(x)\) (u 볼록)로 표현된다. 대응하는 Kantorovich 포텐셜은 다음 형태로 쓸 수 있다:
  
  \[
  \varphi(x) = \|x\|^2 - u(x).
  \]
  
  이 파라미터화는 c-볼록성을 자동으로 보장한다 (섹션 4.2에서 증명). 이를 통해 다루기 어려운 c-볼록 제약을 일반 볼록 제약으로 환원하여 신경망 구조(ICNN)로 근사 가능하게 만든다.

**가정 A. 모멘트와 지지.** \(\mu,\nu\)는 유한한 2차 모멘트를 갖고, 미니배치의 지지는 컴팩트 집합 \(K_\mu, K_\nu\subset\mathbb{R}^d\)에 포함된다.

---

## 4. 수학적 예비 지식

### 4.1 프리멀, 듀얼, 세미듀얼의 도출
**정의 4.1. 프리멀 문제.**
\[
W_2^2(\mu,\nu)=\inf_{\pi\in\Pi(\mu,\nu)} \int\!\|x-y\|^2\,d\pi(x,y).
\]

**정리 4.2. 듀얼 문제.**
\[
W_2^2(\mu,\nu)=\sup_{\varphi,\psi}\Big\{\mathbb{E}_\mu[\varphi]+\mathbb{E}_\nu[\psi]:\ \varphi(x)+\psi(y)\le \|x-y\|^2\ \forall x,y\Big\}.
\]

**증명 개요.** 결합 제약에 라그랑지안을 도입한 뒤 \(\mathcal{P}(\mathbb{R}^d\times\mathbb{R}^d)\)에서 Fenchel–Rockafellar 이중성을 적용한다. 부등식 제약은 비용의 에피그래프 지시함수를 유도하며, 그 지지함수는 점별 제약과 일치한다. 비용의 하반연속성과 2차 모멘트 조건 아래에서 강결합이 성립한다.

**명제 4.3. 세미듀얼.** 최적해에서 \(\psi=\varphi^{\,c}\)가 존재하므로 다음이 성립한다.
\[
W_2^2(\mu,\nu)=\sup_{\varphi}\big\{\mathbb{E}_\mu[\varphi]+\mathbb{E}_\nu[\varphi^{\,c}]\big\}.
\]

**증명.** 듀얼 제약 \(\varphi(x)+\psi(y)\le c(x,y)\)로부터 모든 \(y\)에 대해 \(\psi(y)\le \inf_x\{c(x,y)-\varphi(x)\}=\varphi^{\,c}(y)\)가 된다. 임의의 함수 \(\varphi\)에 대해 \((\varphi,\varphi^{\,c})\)는 듀얼 제약을 만족하므로 가용해이다. \(\psi\)를 \(\varphi^{\,c}\)로 치환하면 목적함수가 증가하므로 최적 \(\psi\)는 \(\varphi^{\,c}\)와 일치한다.

### 4.2 Brenier 파라미터화의 c-볼록성

볼록 \(u\)에 대해 \(\varphi(x)=\|x\|^2-u(x)\)로 두면 다음을 얻는다.
\[
\varphi^{\,c}(y)=\inf_x\{\|x-y\|^2-\|x\|^2+u(x)\}=\|y\|^2-\sup_x\{2\langle x,y\rangle-u(x)\}=\|y\|^2-u^*(2y).
\]

**완전 전개.** \(\|x-y\|^2=\|x\|^2+\|y\|^2-2\langle x,y\rangle\)을 전개해 \(\|x\|^2\)를 상쇄하고, 공액함수 \(u^*(z)=\sup_x\{\langle z,x\rangle-u(x)\}\)의 정의를 \(z=2y\)에 대입한다.

**보조정리 4.4. Fenchel–Young 부등식.** 볼록 함수 \(u\)와 임의의 \(z\)에 대해 \(u(x)+u^*(z)\ge \langle z,x\rangle\)가 Fenchel–Young 부등식으로 알려져 있다. \(z=2y\)와 \(x=y\)를 대입하면 모든 \(y\)에 대해 \(u(y)+u^*(2y)\ge 2\|y\|^2\)를 얻는다. 

**따름정리 4.5. c-볼록 부등식.** \(\varphi(x)=\|x\|^2-u(x)\) 형태에 대해, 보조정리 4.4로부터 \(\varphi^{\,c}(y)=\|y\|^2-u^*(2y)\le -\|y\|^2+u(y)=-\varphi(y)\)가 성립한다.

**비고.** 이 부등식은 \(\varphi=\|\cdot\|^2-u\) 형태일 때만 성립하며, 일반적인 \(\varphi\)에 대해서는 보장되지 않는다. 이후 모든 논의에서 \(\varphi\)는 이 특수 형태를 갖는다고 가정한다.

### 4.3 \(c\)-볼록 부등식과 일관성 갭
정의에 의해 임의의 \(x,y\)에 대해 \(\varphi^{\,c}(y)\le c(x,y)-\varphi(x)\)가 성립한다. 이 부등식으로부터:
- 모든 \(y\)에 대해 성립하므로 \(\inf_y\{c(x,y)-\varphi^{\,c}(y)\}\ge\varphi(x)\), 즉 \(\varphi^{cc}(x)\ge\varphi(x)\)
- 모든 \(x\)에 대해 성립하므로 \((\varphi^{\,c})^{\!c}(y)\ge\varphi^{\,c}(y)\)

이를 바탕으로 다음과 같이 일관성 갭을 정의한다.
\[
\Delta_\mu=\mathbb{E}_\mu[\varphi^{cc}-\varphi]\ge0,\qquad \Delta_\nu=\mathbb{E}_\nu[(\varphi^{\,c})^{\!c}-\varphi^{\,c}]\ge0.
\]

---

## 5. 제안하는 목적함수들
세미듀얼 최적화를 사용하되 \(c\)-볼록 일관성을 다음 함수들로 감시한다.
\[
\begin{aligned}
J_1(\varphi)&=\mathbb{E}_\mu[\varphi]+\mathbb{E}_\nu[\varphi^{\,c}],\\
J_2^{(\mu)}(\varphi)&=\mathbb{E}_\mu[\varphi^{cc}]+\mathbb{E}_\nu[\varphi^{\,c}],\qquad
J_2^{(\nu)}(\varphi)=\mathbb{E}_\mu[\varphi]+\mathbb{E}_\nu[(\varphi^{\,c})^{\!c}],\\
J_3(\varphi)&=\mathbb{E}_\mu[(-\varphi)^{\,c}]+\mathbb{E}_\nu[-\varphi].
\end{aligned}
\]

**명제 5.1. 하한.** 모든 \(\varphi\)에 대해 \(J_1(\varphi)\le W_2^2(\mu,\nu)\)와 \(J_3(\varphi)\le W_2^2(\mu,\nu)\)가 성립한다.

**증명.** 세미듀얼 정의에 의해 \(\sup_\varphi J_1=W_2^2\)이므로 \(J_1\le W_2^2\)이다. 

\(J_3\)의 경우, 거리의 대칭성으로부터 \(W_2^2(\mu,\nu)=W_2^2(\nu,\mu)\)이다. 측도를 바꾼 세미듀얼 문제는
\[
W_2^2(\nu,\mu)=\sup_{\psi}\{\mathbb{E}_\nu[\psi]+\mathbb{E}_\mu[\psi^{\,c}]\}.
\]
이제 \(\varphi=-\psi\)로 치환하면:
\[
\mathbb{E}_\nu[\psi]+\mathbb{E}_\mu[\psi^{\,c}] = \mathbb{E}_\nu[-\varphi]+\mathbb{E}_\mu[(-\varphi)^{\,c}] = J_3(\varphi).
\]
따라서 \(W_2^2(\mu,\nu)=\sup_\varphi J_3(\varphi)\ge J_3(\varphi)\)가 성립한다.

**명제 5.2. 상계 완화.** \(J_1\le J_2^{(\mu)}\)이며 \(J_1\le J_2^{(\nu)}\)이다.

**증명.** \(\varphi\le\varphi^{cc}\)이므로 \(\mathbb{E}_\mu[\varphi]\le\mathbb{E}_\mu[\varphi^{cc}]\)이고, 마찬가지로 \(\mathbb{E}_\nu[\varphi^{\,c}]\le\mathbb{E}_\nu[(\varphi^{\,c})^{\!c}]\)가 성립한다.

**정리 5.3. 일관성에서의 동치.** 다음 세 조건은 서로 동치이다. 첫째, \(\Delta_\mu=\Delta_\nu=0\). 둘째, \(J_1=J_2^{(\mu)}=J_2^{(\nu)}\). 셋째, \(\varphi=\varphi^{cc}\)가 \(\mu\)-a.e. 성립하고 \(\varphi^{\,c}=(\varphi^{\,c})^{\!c}\)가 \(\nu\)-a.e. 성립한다.

**증명.** 첫째에서 둘째는 명제 5.2로부터 즉시 따른다. 둘째에서 셋째는 기대값 부등식에서 등호가 성립하므로 거의 모든 점에서 함수가 일치함을 강제한다. 셋째에서 첫째는 정의로부터 자명하다.

**따름정리 5.4. 타이트니스.** \(\sup_\varphi J_1=\sup_\varphi J_3=W_2^2\). 최적화자 \(\varphi_*\)는 \(c\)-볼록이며 대응 포텐셜은 \(\psi_* = \varphi_*^{\,c}\)이다.

**안정성 주석.** 미니배치에서는 제한된 변환 \(f^{\,c}(\cdot;\eta)=\inf_{x\in\operatorname{supp}(\eta)}\{\|x-\cdot\|^2-f(x)\}\)을 사용한다. 이때도 명제 5.2는 경험적 부등식을 제공하며, 위반은 표본화로 인한 일관성 결함을 의미한다. 이 사실이 스위칭 규칙의 동기를 제공한다.

---

## 6. Back‑and‑Forth 알고리즘

### 6.1 미니배치 c-변환의 한계와 일관성 갭의 필요성

이론적 c-변환은 전체 공간에 대한 infimum이다:
\[
\varphi^{\,c}(y) = \inf_{x\in\mathbb{R}^d} \{\|x-y\|^2 - \varphi(x)\}.
\]

하지만 실제 구현에서는 미니배치 \(\mathcal{B}_x = \{x_1,\ldots,x_n\}\sim\mu\)에 대한 최솟값으로 근사한다:
\[
\varphi^{\,c}(y; \mathcal{B}_x) = \min_{i=1,\ldots,n} \{\|x_i-y\|^2 - \varphi(x_i)\}.
\]

이 근사는 세 가지 문제를 야기한다:

**1. 과대평가 바이어스**: \(\mathcal{B}_x\)가 \(\varphi(x) - \|x-y\|^2\)를 최소화하는 진정한 \(x^*\)를 포함하지 않으면 \(\varphi^{\,c}(y; \mathcal{B}_x) > \varphi^{\,c}(y)\)가 된다.

**2. 배치 의존적 불안정성**: 다른 배치에서는 다른 샘플이 argmin이 되어 \(\varphi^{\,c}\)의 추정치가 변동한다.

**3. 이중 근사 오차**: \(\varphi^{cc}\)를 계산할 때 두 번의 미니배치 근사가 중첩되어 오차가 누적된다:
\[
\varphi^{cc}(x; \mathcal{B}_y, \mathcal{B}_x) = \min_{j} \{\|x-y_j\|^2 - \varphi^{\,c}(y_j; \mathcal{B}_x)\}.
\]

결과적으로 이론적으로 보장되는 \(\varphi = \varphi^{cc}\) (c-볼록성)가 미니배치 환경에서는 깨진다. **일관성 갭 \(\Delta_\mu = \mathbb{E}_\mu[\varphi^{cc} - \varphi]\)는 바로 이 근사 오차를 정량화**하며, 갭이 크다는 것은 현재 \(\varphi_\theta\)가 c-볼록성 제약을 심하게 위반함을 의미한다.

### 6.2 학습 알고리즘

볼록 함수 \(u_\theta\)를 갖는 포텐셜 \(\varphi_\theta=\|\cdot\|^2-u_\theta\)를 ICNN 구조로 근사하고 이를 최적화한다. 반복 \(t\)에서 미니배치 \(X=\{x_i\}_{i=1}^n\sim\mu\)와 \(Y=\{y_j\}_{j=1}^n\sim\nu\)를 추출한 뒤 다음을 계산한다.
\[
\varphi^{\,c}(y_j;X)=\min_i\{\|x_i-y_j\|^2-\varphi(x_i)\},\qquad
\varphi^{cc}(x_i;Y)=\min_j\{\|x_i-y_j\|^2-\varphi^{\,c}(y_j;X)\}.
\]
이 값들로부터 경험적 목적함수 \(\widehat J_1,\widehat J_2^{(\mu)},\widehat J_3\)과 갭 \(\widehat\Delta_\mu,\widehat\Delta_\nu\)를 정의한다.

### 6.3 쌍대성 갭의 선택과 사용

두 갭 \(\Delta_\mu = \mathbb{E}_\mu[\varphi^{cc} - \varphi]\)와 \(\Delta_\nu = \mathbb{E}_\nu[(\varphi^{\,c})^{\!c} - \varphi^{\,c}]\)는 각각 \(\varphi\)와 \(\varphi^{\,c}\)의 c-볼록성 위반 정도를 측정한다. 정리 5.3에 의해 최적해에서는 **둘 다 0**이어야 하므로, 본 연구에서는 다음과 같이 사용한다:

**1. 전환 조건** (알고리즘 1, line 1):
\[
\max\{\widehat\Delta_\mu, \widehat\Delta_\nu\} > \tau
\]

이는 "둘 중 하나라도 큰 갭이 있으면" 갭 감소가 필요함을 의미한다. max를 사용하는 이유는:
- 둘 다 작아야 c-볼록성이 만족됨
- 한쪽만 큰 경우도 포착해야 함
- 보수적 전환 기준 제공

**2. 갭 감소 손실**:
\[
\mathcal{L}_{\text{gap}} = \widehat\Delta_\mu + \widehat\Delta_\nu
\]

합을 사용하는 이유는:
- 두 갭을 동시에 줄이는 것이 목표
- 미분 가능한 단일 목적함수 필요
- 한쪽만 0으로 만드는 것을 방지

**대안적 설계:** \(\max\{\Delta_\mu, \Delta_\nu\}\)만 최소화하면 한쪽 갭을 무시할 위험이 있고, 가중합 \(\alpha\Delta_\mu + (1-\alpha)\Delta_\nu\)는 추가 하이퍼파라미터가 필요하다. 본 연구는 단순성과 효과성의 균형을 위해 조건은 max, 손실은 합을 사용한다.

**알고리즘 1. Back‑and‑Forth 스위칭**
1. \(\max\{\widehat\Delta_\mu,\widehat\Delta_\nu\}>\tau\)이면 갭 감소 단계를 수행한다. 손실 \(\mathcal{L}_{\text{gap}}=\widehat\Delta_\mu+\widehat\Delta_\nu\)을 \(\theta\)에 대해 최소화한다.
2. 그렇지 않으면 하한 상승 단계를 수행한다. 손실 \(\mathcal{L}_{\text{lb}}=-\min\{\widehat J_1,\widehat J_3\}\)을 \(\theta\)에 대해 최소화한다.
3. Armijo 백트래킹이나 코사인 스케줄을 사용하고, 필요하면 log‑sum‑exp 기반의 softmin으로 \(\min\) 연산을 매끄럽게 근사한다.

**명제 6.1. 하강과 상승의 방향.** 위 절차는 Danskin 정리에 의해 타당한 서브그라디언트 갱신을 이룬다. 구체적으로, 각 \(g_i\)가 매끄러운 함수일 때 \(g(y;\theta)=\min_i g_i(y;\theta)\)의 서브그라디언트는 활성 지수 집합 \(I(y;\theta)=\{i: g_i(y;\theta)=g(y;\theta)\}\)에 대해 \(\partial g=\operatorname{co}\{\nabla g_i:i\in I\}\)로 주어진다. 표본에 대해 평균을 취하면 \(\widehat\Delta\)와 \(\widehat J\)의 편향 없는 서브그라디언트를 얻는다. softmin을 사용하면 표준 그래디언트를 사용할 수 있다.

**정리 6.2. 보호된 진행.** 스텝 크기 \(\{\alpha_t\}\)가 \(\sum_t\alpha_t=\infty\)이고 \(\sum_t\alpha_t^2<\infty\)를 만족하면 다음이 성립한다. 첫째, 갭 수열 \(\{\max(\widehat\Delta_\mu^{(t)},\widehat\Delta_\nu^{(t)})\}\)의 하한극한은 0이다. 둘째, 임의의 극한점 \(\theta_*\)는 배치 지지에서 \(c\)-볼록 일관성을 만족한다. 셋째, 갭이 작은 구간에서는 \(\widehat J_1\)이 비감소한다.

**증명 개요.** 갭이 큰 구간에서는 일관성 집합 근방에서의 강제성으로 기대 감소가 보장되고, 갭이 작은 구간에서는 하한 상승이 \(W_2^2\)로 상계되는 준마팅게일형 상승을 유도한다. Robbins-Siegmund 류의 확률근사 이론 결과를 적용하면 수렴 성질이 따른다.

### 6.4 하이퍼파라미터 설정 지침

알고리즘 1은 다음 파라미터를 필요로 한다:

**1. 갭 임계값 \(\tau\)** (핵심):
- 의미: 허용 가능한 c-볼록성 위반 정도
- 설정: \(\tau = 0.01 \times \mathbb{E}[\|x\|^2]\) (데이터 스케일에 비례)
- 민감도: 낮음 (한 자릿수 범위에서 안정)

**2. 동적 가중치 파라미터 k, δ** (선택사항):
- 목적: 갭이 \(\tau\) 근처일 때 부드러운 전환
- 기본값: \(k = 10/\tau\), \(\delta = -\log(0.5)\)
- 대안: \(w = \mathbb{1}[\widehat\Delta > \tau]\) (하드 스위칭)으로 단순화 가능

**3. 기타** (표준):
- 학습률, 배치 크기: 일반적인 신경망 학습과 동일
- 메모리-뱅크 크기 \(M\): 차원 \(d\)의 2~4배 권장
- softmin 온도: 초기 1.0 → 0.1로 annealing

**비교:** WGAN-GP의 gradient penalty 계수 \(\lambda\)는 데이터셋마다 \(10^{-3}\sim 10\) 범위에서 조정이 필요한 반면, 본 방법의 \(\tau\)는 자연스러운 스케일을 가지며 해석이 용이하다.

---

## 7. 메모리‑뱅크 레이어를 갖춘 입력‑볼록 신경망

### 7.1 아키텍처
초기 상태를 \(z_0(x)=x\)로 두고 층 \(\ell=0,\dots,L-1\)에 대해 다음을 계산한다.
\[z_{\ell+1}(x)=\sigma\big(W_\ell^+ z_\ell(x)+U_\ell^+ m(x)+V_\ell x+b_\ell\big).\]
여기서 \(W_\ell^+, U_\ell^+\)는 원소별 비음수이고, \(\sigma\)는 볼록이면서 비감소인 활성함수이며, 메모리‑뱅크 \(m\)은 다음 가운데 하나로 정의한다.
- **Max‑affine**: \(m(x)=\max_k\{\langle a_k,x\rangle+b_k\}\)
- **Log‑sum‑exp**: \(m(x)=\tau\log\sum_k \exp((\langle a_k,x\rangle+b_k)/\tau)\)
- **이차 프로토타입**: \(m(x)=\max_k\{\alpha_k\|x-\kappa_k\|^2+\langle\beta_k,x\rangle+\gamma_k\}\)이며 \(\alpha_k\ge0\)

읽기 연산을 통해 볼록 함수 \(u_\theta(x)=w^{\top} z_L(x)+d\)를 얻고, 포텐셜 \(\varphi_\theta(x)=\|x\|^2-u_\theta(x)\)를 정의한다.

### 7.2 볼록성 보존
**정리 7.1.** 위의 세 가지 \(m\)은 모두 볼록 함수이다. 또한 \(\sigma\)가 볼록이면서 비감소이고 \(W_\ell^+, U_\ell^+\ge0\)이면 모든 \(z_\ell\)이 볼록이므로 \(u_\theta\) 역시 볼록이다.

**증명.** 비음수 선형결합은 볼록성을 보존하고, 선형변환은 볼록이며, 볼록이면서 비감소인 활성함수와의 합성은 볼록성을 유지한다. 귀납법을 적용하면 각 층의 볼록성이 따라온다.

### 7.3 표현력과 안정성
**명제 7.2. 볼록 보편근사.** Max‑affine 계열과 패스스루를 충분히 확장하면 컴팩트 집합에서 임의의 연속 볼록함수를 균일하게 근사할 수 있다.

**명제 7.3. 스무딩 효과.** \(\tau>0\)인 log‑sum‑exp는 최대 연산의 \(C^{\infty}\) 근사이다. 이때 그래디언트의 Lipschitz 상수는 \(O(1/\tau)\)로 제어된다. 따라서 미니배치에서 \(\varphi^{\,c}\) 추정의 분산을 줄이면서 바이어스는 관리 가능하다.

### 7.4 구현 지침
비음수 제약은 softplus나 클램프로 보장하고, 이차 프로토타입의 \(\alpha_k\)는 softplus를 통해 양의 값을 유지시킨다. 온도 \(\tau\)는 크게 시작해 점차 줄이는 방식으로 조정한다. 슬롯 파라미터에는 \(\ell_2\) 정규화나 드롭아웃을 적용한다. softmin을 사용하면 변환의 경사를 자동미분으로 안정적으로 전파할 수 있다.

---

## 8. 오차 제어: 포텐셜 근사에서 거리 바이어스로

컴팩트 집합 \(K\)가 \(\operatorname{supp}(\mu)\cup\operatorname{supp}(\nu)\)를 포함하고, 참 볼록 함수 \(u_*\)에 대한 근사 \(u_\theta\)의 균일 오차가 \(\varepsilon\)이라고 하자: \(\sup_{x\in K}|u_\theta(x)-u_*(x)|\le\varepsilon\).

**정리 8.1. 거리 추정 오차 상계.**
포텐셜 \(\varphi_\theta=\|\cdot\|^2-u_\theta\)에 대해 다음 부등식이 성립한다:
\[
0\le W_2^2(\mu,\nu)-J_1(\varphi_\theta)\le C(K,\mu,\nu)\,\varepsilon,
\]
여기서 \(C(K,\mu,\nu)\)는 집합 \(K\)의 직경과 분포의 2차 모멘트에 의존하는 상수이다.

**증명 개요.** 
1. \(u\)와 \(\tilde u\)의 균일 오차가 \(\varepsilon\)일 때, c-변환의 점별 오차를 추정한다. 섹션 4.2의 식으로부터:
\[
|\varphi^{\,c}(y;u) - \varphi^{\,c}(y;\tilde u)| = |u^*(2y) - \tilde u^*(2y)|.
\]

2. 볼록 공액의 안정성: 균일 오차 \(\varepsilon\)에 대해 Legendre 변환의 Lipschitz 연속성으로부터 \(|u^*(z) - \tilde u^*(z)| \le \varepsilon\)를 얻는다 (컴팩트 집합에서).

3. 따라서 \(|\varphi^{\,c}(y;u) - \varphi^{\,c}(y;\tilde u)| \le \varepsilon\)가 성립한다.

4. 세미듀얼 범함수에 대해:
\[
|J_1(\varphi_u) - J_1(\varphi_{\tilde u})
