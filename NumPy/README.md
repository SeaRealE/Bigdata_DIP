# NumPy 
#### Numerical Python
: 과학 계산을 위한 대부분의 패키지는 NumPy의 배열 객체를 데이터 교환을 위한 공통 언어처럼 사용한다.

#### NumPy에서 제공하는 것들
- 빠른 배열 계산과 유연한 브로드캐스팅 기능
- 반복문을 작성할 필요없이 전체 데이터 배열을 빠르게 계산가능
- 배열 데이터를 디스크에 쓰거나 읽을 수 있는 도구와 메모리에 적재된 파일을 다루는 도구
- 선형대수, 난수 생성기, 푸리에 변환 가능
- C, C++, 포트란으로 작성한 코드를 연결할 수 있는 C API

#### 대용량 데이터 배열을 효율적으로 다루도록 설계됨
- 데이터를 다른 내장 Python 객체와 구분되는 연속된 메모리 블록에 저장
- 각종 알고리즘은 모두 C로 작성되어 타입 검사나 다른 오버헤드없이 메모리를 직접 조작
- 내장 Python의 연속된 자료형들보다 훨씬 더 적은 메모리 사용
- 연산은 Python 반복문을 사용하지 않고 전체배열에 대한 복잡한 계산을 수행

----
## 기본 설정  
```{.python}
import numpy as np
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc('figure',figsize=(10,6))
np.set_printoptions(precision=4, suppress=True)
```
## 내용
- Numpy ndarray : 다차원 배열 객체
- 유니버셜 함수 : 배열의 각 원소를 빠르게 처리하는 함수
- 배열을 이용한 배열 지향 프로그래밍
