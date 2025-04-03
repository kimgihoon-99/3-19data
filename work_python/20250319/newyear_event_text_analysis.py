# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 08:52:06 2025

@author: Admin

"""
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### 수집된 파일 로드 ###
df = pd.read_csv("./event_text.csv")
df.head()
df.tail()

### 데이터 전처리 ###
# 중복된 글 제거 
'''
온라인으로 수집된 데이터는 다양한 이유로 중복 생성
   웹사이트에서 전송 버튼을 여러 번 누르거나,
   새로 고침을 하거나
   네트워크나 UX 관련 오류 문제가 발생

=> 중복 데이터가 있으면 빈도 분석에 제대로 되지 않기 때문에 중복 입력값을 제거
   drop_duplicates() : 전체 열의 중복을 제거
   keep= "last" / "first" / False
'''
print(df.shape)   #  (2449, 1)

df = df.drop_duplicates(['text'], keep='last')
print(df.shape)  #  (2410, 1)

df['origin_text'] = df['text']

# 소문자 변환 
df['text'] = df['text'].str.lower()

# 같은 의미의 단어를 하나로 통일
# 예) python => 파이썬
df['text'] = df['text'].str.replace('python', '파이썬')  \
                       .str.replace('pandas', '판다스')  \
                       .str.replace('javascript', '자바스크립트')  \
                       .str.replace('java', '자바')  \
                       .str.replace('react', '리엑트')

### 문자열 분리로 관심 강의 분리 ###
'''
이 이벤트에는 "관심강의"라는 텍스트가 있음.
"관심강의"를 기준으로 텍스트를 분리하고 관심강의 뒤에 있는 텍스트를 가져옴
=> 대부분 "관심강의"라는 텍스트를 쓰고 뒤에 강의명을 쓰기 때문

전처리한 내용은 'course' 라는 새로운 컬럼에 담고
'관심 강의', '관심 강좌'에 대해서도 똑같이 전처리.

":" 특수문자를 빈문자로 변경
'''
df['course'] = df['text'].apply(lambda x: x.split('관심강의')[-1])
df['course'] = df['course'].apply(lambda x: x.split('관심 강의')[-1])
df['course'] = df['course'].apply(lambda x: x.split('관심 강좌')[-1])
df['course'] = df['course'].str.replace(":", "")

### 텍스트에서 특정 키워드를 추출 ###
# 띄어 쓰기를 제거한 텍스트에서 키워드 추출
search_keyword = ['머신러닝', '딥러닝', '파이썬', '판다스', '공공데이터',
                  'django', '크롤링', '시각화', '데이터분석', 
                  '웹개발', '엑셀', 'c', '자바', '자바스크립트', 
                  'node', 'vue', '리액트']

# 키워드가 있는지 여부를 True, False값
for keyword in search_keyword:
    df[keyword] = df["course"].str.contains(keyword)

df.head()

# 파이썬|공공데이터|판다스 라는 텍스트가 들어가는 데이터가 있는지
df_python = df[df["text"].str.contains("파이썬|공공데이터|판다스")].copy()
df_python.shape       # (429, 20)

# 해당 키워드의 등장 빈도수
df[search_keyword].sum().sort_values(ascending=False)

# 공공데이터 텍스트가 들어가는 문장만 찾음
text = df.loc[(df['공공데이터'] == True), "text"]

for t in text:
    print("-"*20)
    print(t)

### 빈도수 계산을 위한 텍스트 데이터 벡터화 ###
# BOW 단어 가방에 단어를 토큰화 해서 담아줌 
# CountVectorizer()를 통해 벡터화
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer= 'word',   # 캐릭터 단위로 벡터화 할 수도 있음
                             tokenizer= None,    # 토크나이저를 따로 지정해 줄 수도 있음
                             preprocessor= None, # 전처리 도구
                             stop_words= None,   # 불용어 nltk등의 도구를 사용할 수도 있음
                             min_df =2,          # 토큰이 나타날 최소 문서 개수
                             ngram_range=(3, 6), # BOW의 단위 갯수의 범위를 지정
                             max_features= 2000  # 만들 피치의 수, 단어의 수
                             )

feature_vector = vectorizer.fit_transform(df['course'])
feature_vector.shape  # (2410, 2000)

vocab = vectorizer.get_feature_names_out()
vocab[:10]

pd.DataFrame(feature_vector[:10].toarray(), columns=vocab).head()
'''
전체 단어 가방에서 해당 리뷰마다 등장하는 단어에 대한 빈도수

행은 각 리뷰를 의미 / 0은 등장하지 않는다

단어 가방으로 벡터화하면 희소 행렬이 만들어지는 단점! 
희소 행렬(Sparse Matrix) : 행렬의 대부분 원소가 0인 행렬을 의미

희소 행렬의 활용 분야
   웹 검색 엔진: 웹 페이지 간의 연결 관계를 나타내는 행렬은 희소 행렬의 형태
   소셜 네트워크: 사용자 간의 연결 관계를 나타내는 행렬은 희소 행렬의 형태
   추천 시스템: 사용자-아이템 간의 평점 데이터를 나타내는 행렬은 희소 행렬의 형태
   자연어 처리: 문서-단어 간의 출현 빈도를 나타내는 행렬은 희소 행렬의 형태

대부분의 원소가 0: 희소 행렬은 전체 원소 중에서 0이 아닌 원소의 비율이 매우 낮다

저장 공간 효율성: 일반적인 2차원 배열로 희소 행렬을 저장하면 많은 공간이 낭비되므로,
                 0이 아닌 원소만 저장하는 효율적인 저장 방식이 필요

연산 속도 향상: 희소 행렬의 특성을 활용하여 0이 아닌 원소에 대해서만 연산을 수행하면
               연산 속도를 크게 향상시킬 수 있다
               
=> 희소 행렬은 다양한 분야에서 대용량 데이터를 효율적으로 처리하는 데 중요한 역할!!!
'''
# 단어가 전체에서 등장하는 횟수 
dist = np.sum(feature_vector, axis= 0)

df_freq = pd.DataFrame(dist, columns=vocab)
df_freq.T.sort_values(by=0, ascending=False).head(30)

### 중복을 처리 ### 
df_freq_T = df_freq.T.reset_index()
df_freq_T.columns = ["course", "freq"]
'''
중복을 제거하기 위해 강의명에서 지식공유자의 이름(***)을 빈 문자열로 변경

Lambda 식을 사용해서 강의명을 x.split()으로 나눈 다음
[:4], 즉 앞에서 4개까지만 텍스트를 가져오고 다시 join으로 합친다.
=> 중복된 텍스트를 구분해서 보기 위함!

빈도수를 기준으로 내림차순으로 10개를 미리 보기로 확인
'''
df_freq_T['course_find'] = df_freq_T['course'].str.replace('박조은', "")
df_freq_T['course_find'] = df_freq_T['course_find'].apply(lambda x : " ".join(x.split()[:4]))

df_freq_T.sort_values(["course_find", "freq"], ascending=False).head(10)
'''
3개의 ngram과 빈도수로 역순 정렬을 하게 되면 빈도수가 높고
=> ngram수가 많은 순으로 정렬이 됨

drop_duplicates로 첫 번째 강좌를 남기고 나머지 중복을 삭제
'''
print(df_freq_T.shape)     # (2000, 3)

df_course = df_freq_T.drop_duplicates(["course_find", "freq"], keep="first")
print(df_course.shape)    # (1442, 3)

# 빈도수로 정렬을 하고, 어떤 강좌가 댓글에서 가장 많이 언급되었는지 확인
df_course = df_course.sort_values(by="freq", ascending=False)
df_course.head(20)

# 전처리가 다 되었다면, 다른 팀 또는 담당자에게 전달하기 위해 csv 형태로 저장
df_course.to_csv("./event_course_name_freq.csv")

### TF_IDF로 가중치를 주어 벡터화 ###
'''
TfidfTransformer()

TfidfTransformer()
   CountVectorizer()와 TfidfTransformer()를 한 번에 처리.
   
CountVectorizer()와 TfidfTransformer()를 따로 사용
=> 빈도 기반의 단어 가방과 TF-IDF 가중치를 적용한 값을 비교
'''
from sklearn.feature_extraction.text import TfidfTransformer

tfidftrans = TfidfTransformer(smooth_idf=False)
feature_tfidf = tfidftrans.fit_transform(feature_vector)
feature_tfidf.shape   # (2410, 2000)

# 각 row에서 전체 단어 각방 모형에 등장하는 단어에 대한 
# one-hot-vector에 TF-IDF 가중치를 반영한 결과
# => feature_tfidf.toarray()로 배열을 만든 뒤 확인!
tfidf_freq = pd.DataFrame(feature_tfidf.toarray(), columns=vocab)
tfidf_freq.head()
'''
sum()으로 tfidf_freq의 합계 
이유 : TF-IDF 가중치를 적용하더라도 희소한 행렬이 만들어지기 때문
      각 피처마다 가중치가 제대로 작용됐는지 확인하기 위해
'''
df_tfidf = pd.DataFrame(tfidf_freq.sum())
df_tfidf_top = df_tfidf.sort_values(by=0, ascending=False)
df_tfidf_top.head(10)

##-----------------군집화 ----------------------------------------##
'''
KMeans : 머신러닝의 비지도학습 기법 중 하나
주어진 데이터를 K개로 묶는 알고리즘 
군집 간의 거리 차이의 분산을 최소화하는 방식으로 군집

데이터 집합에서 K개의 데이터 개체를 임의로 추출하고
각 클러스터의 중심점(centroid)을 초깃값으로 설정

K개의 군집과 데이터 집합의 개체의 거리를 구해
각 개체가 어느 중심정과 가장 유사도가 높은지를 계산
찾은 중심점으로 다시 데이터 군집의 중심점을 계산하는 방법은 반복
유클리드 거리 측정 방법

K-평균 알고리즘의 특징
  알고리즘이 단순하고 구현이 쉽다
  대용량 데이터에도 비교적 빠르게 처리할 수 있다
  
  클러스터의 개수 K개를 미리 지정
  초기 중심 설정에 따라 결과가 달라질 수 있다
  클러스터의 모양이 원에 가까울 때 효과적
  이상치에 민감
  
K-평균 알고리즘의 활용 분야
  이미지 분할: 이미지를 여러 개의 영역으로 분할하는데 사용
  고객 분류: 고객을 여러 그룹으로 나누어 마케팅 전략을 수립하는데 사용
  문서 클러스터링: 문서를 주제별로 분류하는 데 사용
  이상 감지: 정상 데이터와 거리가 먼 이상 데이터를 탐지하는데 사용
  
K-평균 알고리즘 사용 시 고려사항
  적절한 k 값 선택: 엘보우 방법, 실루엣 분석 등 다양한 방법을 사용하여 최적의 k 값을 선택
  초기 중심 설정: k-means++ 알고리즘과 같은 초기 중심 설정 방법을 사용하여 초기 중심 설정의 
  영향을 줄 수 있다
  
  데이터 전처리: 데이터의 스케일링, 이상치 제거 등 전처리 과정을 통해
                알고리즘의 성능을 향상시킬 수 있다!!!!!
                
K-평균 알고리즘의 작동 방식
1. 초기 중심 설정: k개의 클러스터 중심을 임의로 설정
2. 클러스터 할당: 각 데이터 포인트를 가장 가까운 클러스터 중심에 할당
3. 중심 재계산: 각 클러스터의 중심을 해당 클러스터에 속한 데이터 포인트들의 평균으로 다시 계산
4. 반복: 2번과 3번 과정을 클러스터 중심이 더 이상 변하지 않거나 최대 반복 횟수에 도달할 때까지 반복
'''

from sklearn.cluster import KMeans     
from tqdm import trange

# 엘보우 방법
inertia = []    
start = 10
end = 70

 
for i in trange(start, end):            
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(feature_tfidf)
    inertia.append(kmeans.inertia_)

'''
엘보우(Elbow) 방법
   이너셔 값이 급격하게 꺽이는 지점을 찾아 군집으로 정하는 것
   현실세계 대이터를 다루다 보면, 이론처럼 급격하게 꺾이는 지점이 나타나지 않을 수 있다.
   클러스터의 수가 너무 많으먼 군집화 값을 관리하기 어려울 수도 있다.

'''
# inertia 값을 시각화
# x축에는 클러스터의 수 / y축에는 inertia 값
plt.plot(range(start, end), inertia)
plt.title("KMeans 클러스터 수 비교")
plt.show()


n_clusters = 50
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(feature_tfidf)

prediction = kmeans.predict(feature_tfidf)
df["cluster"] = prediction

df["cluster"].value_counts().head(10)
'''
value_counts() : 시리즈(series) 유일값의 빈도를 계산해 주는 함수
cluster
0     1749
7       44
4       41
5       30
22      27
10      26
48      26
14      26
3       25
1       23
Name: count, dtype: int64
'''
### MiniBatchKMeans ###
'''
데이터가 없다면 군집화에 속도가 오래 걸린다.
배치 사이즈를 지정해서 군집화를 진행하면 조금 빠르게 작업.

1. 적절한 클러슽의 개수를 알기 위해 이니셔 값을 구한다.
'''
from sklearn.cluster import MiniBatchKMeans

from sklearn.metrics import silhouette_score

b_inertia = []
silhouettes = []
 
for i in trange(start, end):
    mkmeans = MiniBatchKMeans(n_clusters=i, random_state=42)
    mkmeans.fit(feature_tfidf)
    b_inertia.append(mkmeans.inertia_)
    silhouettes.append(silhouette_score(feature_tfidf, mkmeans.labels_))

plt.plot(range(start, end), b_inertia)
plt.title("MiniBatchKMeans 클러스터 수 비교")
plt.show()
'''
이너셔 값을 시각화하는 이유 : 엘보 기법을 사용

y축의 이니셔 값 : 가장 가까운 군집의 중심과 샘플의 거리 제곱합
                 이너셔 값이 작을수록 군집이 잘 되었다고 평가
                 => 중심과 샘플의 거리가 가까울수록 군집이 잘 되었다고 볼 수 있기 때문
                 
반복문을 통해 이너셔 값을 구할 때
  첫 군집은 랜덤하게 군집의 중심을 설정하기 때문
  => 높은 값이 나올 수 밖에 없다.

  다음 군집부터는 거리를 계산해 군집의 중심을 이동하므로
  중심과 군집은 점점 가까워지고 값이 점점 낮아지는 그래프가 그려진다.

전처리가 잘 된 정형 데이터라면 팔꿈치처럼 급격하게 꺾이는 부분이 이상적으로 등장할 수 있지만
비정형 텍스트 데이터에서는 이론처럼 급격하게 꺾이는 부분은 나타나지 않는다.
'''

'''
실루엣 점수란?
클러스터링된 데이터가 얼마나 잘 분리되었는지를 나타내는 지표

a: 해당 데이터 포인트와 동일한 클러스터 내의 다른 모든 데이터 포인트 간의 평균 거리
   클러스터 내 응집도
   
b: 해당 데이터 포인트와 가장 가까운 다른 클러스터의 모든 데이터 포인트 간의 평균 거리
   클러스터 간 분리도

silhouette_score()
  X: 클러스터링된 데이터 포인트의 특징 행렬 (Numpy 배열 또는 희소 행렬)
  Labels: 각 데이터 포인트의 클러스터 레이블 (Numpy 배열)
  metric: 거리 측정 방법 (기본값은 ' euclidean')
  sample_size: 실루엣 점수를 계산할 샘플 크기 (기본값은 None, 즉 모든 데이터 포인트 사용)
  random_state: 샘플링 시 난수 생성 시드
  
실루엣 점수 활용
  최적의 클러스터 개수 선택: 다양한 클러스터 개수에 대해 실루엣 점수를 계산하고,
                            가장 높은 점수를 가지는 클러스터 개수를 선택
                            
    
  클러스터링 알고리즘 성능 평가: 다양한 클러스터링 알고리즘의 성능을 비교하는데 사용
  
  클러스터링 결과 시각화: 실루엣 플롯을 통해 각 클러스터의 응집도와 분리도를 시각적으로 확인
'''

### yellowbrick 은 머신러닝 시각화 도구 ###
# pip install yellowbrick

from yellowbrick.cluster import KElbowVisualizer
 
KElbowM = KElbowVisualizer(kmeans, k=70)
KElbowM.fit(feature_tfidf.toarray())
KElbowM.show()

#------------------- 군집 결과 시각화  -----------------#
mkmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
mkmeans.fit(feature_tfidf)

prediction = mkmeans.predict(feature_tfidf)
df["bcluster"] = prediction

df["bcluster"].value_counts().head(10)
'''
bcluster
40    1112
0      211
2      148
9       87
42      59
8       50
22      49
44      37
13      34
26      32
Name: count, dtype: int64
'''
df.loc[df["bcluster"] == 21, ["bcluster", "cluster", "course"]]

df.loc[df["bcluster"] == 24, ["bcluster", "cluster", "origin_text", "course"]].tail(10)

### 클러스터 예측 평가 ###
'''
분류, 회귀 모델은 지도학습으로, 정답이 있기 때문에 
정답과 예측값을 비교해 볼 수 있다.

군집화는 비지도학습으로, 정답이 없기 때문에
목적에 따라 평가 방법을 정해야 한다.

클러스터의 예측 정확도를 확인
=> n_clusters는 위에서 정의한 클러스터 수를 사용
'''
feature_array = feature_vector.toarray()
'''
 unique(): 예측한 클러스터의 유일값을 구한다.
 where(prediction==label): 예측한 값이 클러스터 번호와 일치하는 것을 가져온다.
 mean(): 클러스터의 평균 값을 구한다.
 np.argsort(x_means)[::-1][:n_clusters]: 값을 역순으로 정렬해서 클러스터 수만큼 가져온다.
'''
labels = np.unique(prediction)

df_cluster_score = []
df_cluster = []

for label in labels:
    id_temp = np.where(prediction==label)
    x_means = np.mean(feature_array[id_temp], axis = 0) 
    sorted_means = np.argsort(x_means)[::-1][:n_clusters]
    
    features = vectorizer.get_feature_names_out()
    best_features = [(features[i], x_means[i]) for i in sorted_means] 
    # 클러스터별 전체 스코어를 구한다.
    df_score = pd.DataFrame(best_features, columns = ['features', 'score'])
    df_cluster_score.append(df_score)
    # 클러스터 대표 키워드
    df_cluster.append(best_features[0])

'''
점수가 클수록 예측 정확도가 높다
MiniBatchKMeans로 예측한 값을 기준으로 정렬해, 각  클러스터에서 점수가 높은 단어를 추출
'''
pd.DataFrame(df_cluster, columns= ['features', 'score'])  \
             .sort_values(by=['features', 'score'], ascending=False)

# score 정확도가 1이 나온 클러스터
df.loc[df['bcluster']== 28, ['bcluster', 'cluster', 'course']]

from yellowbrick.cluster import SilhouetteVisualizer

visualizer = SilhouetteVisualizer(mkmeans, colors='yellowbrick')

visualizer.fit(feature_tfidf.toarray())
visualizer.show()
'''
34 번 군집에 너무 많은 데이터가 몰려 있고,
군집의 실루엣 개수가 음수로 나타나는 것들도 있다.
=> 다른 군집과 겹치는 부분이 많아서 제대로 군집되지 않았음

비정형 텍스트 데이터를 군집화할 때는
=> 전처리, 정규화, 벡터화 등을 고려
'''
from sklearn.datasets import make_blobs

# 예시 데이터 생성
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

# K-평균 클러스터링 모델 생성 및 학습
model = KMeans(n_clusters=4, random_state=42)

visualizer = SilhouetteVisualizer(model, colors='yellowbrick')

visualizer.fit(X)

visualizer.poof()

plt.show()

























































































































































































