#!/usr/bin/env python
# coding: utf-8

# **노트**: 이 주피터 노트북의 결과가 책에 있는 것과 조금 다를 수 있습니다. 대부분은 훈련 알고리즘들이 가지고 있는 무작위성 때문입니다. 가능하면 노트북의 결과를 동일하게 유지하려고 하지만 모든 플랫폼에서 동일한 출력을 낸다고 보장하긴 어렵습니다. 어떤 데이터 구조(가령 딕셔너리)는 아이템의 순서가 일정하지 않습니다. 마지막으로 몇 가지 사소한 버그 수정(해당 부분에 설명을 추가했습니다) 때문에 결과가 조금 달라졌습니다. 하지만 책에서 제시한 설명은 유효합니다.

# # 설정

# 파이썬 2와 3을 모두 지원합니다. 공통 모듈을 임포트하고 맷플롯립 그림이 노트북 안에 포함되도록 설정하고 생성한 그림을 저장하기 위한 함수를 준비합니다:

# In[1]:


# 파이썬 2와 파이썬 3 지원
from __future__ import division, print_function, unicode_literals

# 공통
import numpy as np
import os

# 일관된 출력을 위해 유사난수 초기화
np.random.seed(42)

# 맷플롯립 설정
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 한글출력
matplotlib.rc('font', family='NanumBarunGothic')
plt.rcParams['axes.unicode_minus'] = False

# 그림을 저장할 폴드
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# # 데이터 다운로드

# In[2]:


import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# In[3]:


fetch_housing_data()


# In[4]:


import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# In[5]:


housing = load_housing_data()
housing.head()


# In[6]:


housing.info()


# In[7]:


housing["ocean_proximity"].value_counts()


# In[8]:


housing.describe()


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()


# In[10]:


# 일관된 출력을 위해 유사난수 초기화
np.random.seed(42)


# In[11]:


import numpy as np

# 예시를 위해서 만든 것입니다. 사이킷런에는 train_test_split() 함수가 있습니다.
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[12]:


train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")


# In[13]:


from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# 위의 `test_set_check()` 함수는 파이썬 2와 파이썬 3에서 모두 작동되고 다음의 hashlib를 사용한 구현보다 훨씬 빠릅니다.

# In[14]:


import hashlib

def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return bytearray(hash(np.int64(identifier)).digest())[-1] < 256 * test_ratio


# In[15]:


# 이 버전의 test_set_check() 함수가 파이썬 2도 지원합니다.
def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return bytearray(hash(np.int64(identifier)).digest())[-1] < 256 * test_ratio


# In[16]:


housing_with_id = housing.reset_index()   # `index` 열이 추가된 데이터프레임이 반환됩니다.
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")


# In[17]:


housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")


# In[18]:


test_set.head()


# In[19]:


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# In[20]:


test_set.head()


# In[21]:


housing["median_income"].hist()


# In[22]:


# 소득 카테고리 개수를 제한하기 위해 1.5로 나눕니다.
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
# 5 이상은 5로 레이블합니다.
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)


# In[23]:


housing["income_cat"].value_counts()


# In[24]:


housing["income_cat"].hist()
save_fig('income_category_hist')


# In[25]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[26]:


strat_test_set["income_cat"].value_counts() / len(strat_test_set)


# In[27]:


housing["income_cat"].value_counts() / len(housing)


# In[28]:


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100


# In[29]:


compare_props


# In[30]:


for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# # 데이터 이해를 위한 탐색과 시각화

# In[31]:


housing = strat_train_set.copy()


# In[32]:


ax = housing.plot(kind="scatter", x="longitude", y="latitude")
ax.set(xlabel='경도', ylabel='위도')
save_fig("bad_visualization_plot")


# In[33]:


ax = housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
ax.set(xlabel='경도', ylabel='위도')
save_fig("better_visualization_plot")


# `sharex=False` 매개변수는 x-축의 값과 범례를 표시하지 못하는 버그를 수정합니다. 이는 임시 방편입니다(https://github.com/pandas-dev/pandas/issues/10611 참조). 수정 사항을 알려준 Wilmer Arellano에게 감사합니다.

# In[34]:


ax = housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="인구", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
ax.set(xlabel='경도', ylabel='위도')
plt.legend()
save_fig("housing_prices_scatterplot")


# In[35]:


import matplotlib.image as mpimg
california_img=mpimg.imread(PROJECT_ROOT_DIR + '/images/end_to_end_project/california.png')
ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
                       s=housing['population']/100, label="인구",
                       c="median_house_value", cmap=plt.get_cmap("jet"),
                       colorbar=False, alpha=0.4,
                      )
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5)
plt.ylabel("위도", fontsize=14)
plt.xlabel("경도", fontsize=14)

prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar()
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label('중간 주택 가격', fontsize=16)

plt.legend(fontsize=16)
save_fig("california_housing_prices_plot")
plt.show()


# In[36]:


corr_matrix = housing.corr()


# In[37]:


corr_matrix["median_house_value"].sort_values(ascending=False)


# In[38]:


from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
save_fig("scatter_matrix_plot")


# In[39]:


housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
plt.axis([0, 16, 0, 550000])
save_fig("income_vs_house_value_scatterplot")


# In[40]:


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


# In[41]:


corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[42]:


housing.plot(kind="scatter", x="rooms_per_household", y="median_house_value",
             alpha=0.2)
plt.axis([0, 5, 0, 520000])
plt.show()


# In[43]:


housing.describe()


# # 머신러닝 알고리즘을 위한 데이터 준비

# In[44]:


housing = strat_train_set.drop("median_house_value", axis=1) # 훈련 세트를 위해 레이블 삭제
housing_labels = strat_train_set["median_house_value"].copy()


# In[46]:


housing.head() # 레이블 제거된 데이터


# In[47]:


housing_labels.head() # 레이블 데이터


# In[48]:


sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows


# In[49]:


sample_incomplete_rows.dropna(subset=["total_bedrooms"])    # 옵션 1


# In[50]:


sample_incomplete_rows.drop("total_bedrooms", axis=1)       # 옵션 2


# In[51]:


median = housing["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # 옵션 3
sample_incomplete_rows


# `sklearn.preprocessing.Imputer` 클래스는 사이킷런 0.20 버전에서 사용 중지 경고가 발생하고 0.22 버전에서 삭제될 예정입니다. 대신 추가된 `sklearn.impute.SimpleImputer` 클래스를 사용합니다.

# In[52]:


#from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")


# 중간값이 수치형 특성에서만 계산될 수 있기 때문에 텍스트 특성을 삭제합니다:

# In[53]:


housing_num = housing.drop('ocean_proximity', axis=1)
# 다른 방법: housing_num = housing.select_dtypes(include=[np.number])


# In[54]:


imputer.fit(housing_num)


# In[55]:


imputer.statistics_


# 각 특성의 중간 값이 수동으로 계산한 것과 같은지 확인해 보세요:

# In[56]:


housing_num.median().values


# 훈련 세트 변환:

# In[57]:


X = imputer.transform(housing_num)


# In[58]:


housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index = list(housing.index.values))


# In[59]:


housing_tr.loc[sample_incomplete_rows.index.values]


# In[60]:


imputer.strategy


# In[61]:


housing_tr = pd.DataFrame(X, columns=housing_num.columns)
housing_tr.head()


# 이제 범주형 입력 특성인 `ocean_proximity`을 전처리합니다:

# ### 책에 실린 방법

# In[62]:


housing_cat = housing['ocean_proximity']
housing_cat.head(10)


# 판다스의 `factorize()` 메소드는 문자열 범주형 특성을 머신러닝 알고리즘이 다루기 쉬운 숫자 범주형 특성으로 변환시켜 줍니다:

# In[63]:


housing_cat_encoded, housing_categories = housing_cat.factorize()
housing_cat_encoded[:10]


# In[64]:


housing_categories


# `OneHotEncoder`를 사용하여 범주형 값을 원-핫 벡터로 변경합니다:

# 사이킷런 0.20 버전에서 OneHotEncoder의 동작 방식이 변경되었습니다. 종전에는 0~최댓값 사이의 정수를 카테고리로 인식했지만 앞으로는 정수나 문자열에 상관없이 고유한 값만을 카테고리로 인식합니다. 경고 메세지를 피하기 위해 `categories` 매개변수를 `auto`로 설정합니다.

# In[65]:


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(categories='auto')
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot


# `OneHotEncoder`는 기본적으로 희소 행렬을 반환합니다. 필요하면 밀집 배열로 변환할 수 있습니다:

# In[66]:


housing_cat_1hot.toarray()


# `CategoricalEncoder`는 하나 이상의 특성을 가진 2D 배열을 기대합니다. 따라서 `housing_cat`을 2D 배열로 바꾸어 주어야 합니다:

# In[67]:


#from sklearn.preprocessing import CategoricalEncoder # Scikit-Learn 0.20에서 추가 예정

cat_encoder = CategoricalEncoder()
housing_cat_reshaped = housing_cat.values.reshape(-1, 1)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat_reshaped)
housing_cat_1hot


# 사이킷런 0.20 개발 브랜치에 있던 `CategoricalEncoder`는 새로운 `OneHotEncoder`와 `OrdinalEncoder`로 나뉘었습니다. `OneHotEncoder`로 문자열로 된 범주형 변수도 변환할 수 있습니다:

# In[68]:


from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder(categories='auto')
housing_cat_reshaped = housing_cat.values.reshape(-1, 1)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat_reshaped)
housing_cat_1hot


# 기본 인코딩은 원-핫 벡터이고 희소 행렬로 반환됩니다. `toarray()` 메소드를 사용하여 밀집 배열로 바꿀 수 있습니다:

# In[69]:


housing_cat_1hot.toarray()


# 또는 encoding 매개변수를 `"onehot-dense"`로 지정하여 희소 행렬대신 밀집 행렬을 얻을 수 있습니다. 0.20 버전의 `OneHotEncoder`는 `sparse=Fasle` 옵션을 주어 밀집 행렬을 얻을 수 있습니다:

# In[70]:


# cat_encoder = CategoricalEncoder(encoding="onehot-dense")
cat_encoder = OneHotEncoder(categories='auto', sparse=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat_reshaped)
housing_cat_1hot


# In[71]:


cat_encoder.categories_


# ### future_encoders.py를 사용한 새로운 방법

# In[72]:


housing_cat = housing[['ocean_proximity']]
housing_cat.head(10)


# 주의: 번역서는 판다스의 `Series.factorize()` 메서드를 사용하여 문자열 범주형 특성을 정수로 인코딩합니다. 사이킷런 0.20에 추가될 `OrdinalEncoder` 클래스(PR #10521)는 입력 특성(레이블 `y`가 아니라 `X`)을 위해 설계되었고 파이프라인(나중에 이 노트북에서 나옵니다)과 잘 작동되기 때문에 더 좋은 방법입니다. 지금은 `future_encoders.py` 파일에서 임포트하지만 사이킷런 0.20 버전이 릴리스되면 `sklearn.preprocessing`에서 바로 임포팅할 수 있습니다.
# 
# 0.20 버전 릴리스에 맞추어 `sklearn.preprocessing`에서 임포트합니다.

# In[73]:


# from future_encoders import OrdinalEncoder
from sklearn.preprocessing import OrdinalEncoder


# In[74]:


ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]


# In[75]:


ordinal_encoder.categories_


# 주의: 번역서는 `CategoricalEncoder`를 사용하여 각 범주형 값을 원-핫 벡터로 변경합니다. 이 클래스는 `OrdinalEncoder`와 새로운 `OneHotEncoder`로 리팩토링되었습니다. 지금은 `OneHotEncoder`가 정수형 범주 입력만 다룰 수 있지만 사이킷런 0.20에서는 문자열 범주 입력도 다룰 수 있을 것입니다(PR #10521). 지금은 `future_encoders.py` 파일에서 임포트하지만 사이킷런 0.20 버전이 릴리스되면 `sklearn.preprocessing`에서 바로 임포팅할 수 있습니다.
# 
# 0.20 버전 릴리스에 맞추어 `sklearn.preprocessing`에서 임포트합니다(사실 우리는 이미 위에서 0.20 버전의 `OneHotEncoder`를 사용했습니다).

# In[76]:


# from future_encoders import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder(categories='auto')
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot


# 기본적으로 `OneHotEncoder` 클래스는 희소 행렬을 반환하지만 필요하면 `toarray()` 메서드를 호출하여 밀집 배열로 바꿀 수 있습니다:

# In[77]:


housing_cat_1hot.toarray()


# 또는 `OneHotEncoder` 객체를 만들 때 `sparse=False`로 지정하면 됩니다:

# In[78]:


cat_encoder = OneHotEncoder(categories='auto', sparse=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot


# In[79]:


cat_encoder.categories_


# ### 다시 책의 내용이 이어집니다

# 추가 특성을 위해 나만의 변환기를 만들겠습니다:

# In[79]:


from sklearn.base import BaseEstimator, TransformerMixin

# 컬럼 인덱스
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# In[80]:


housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs, 
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"])
housing_extra_attribs.head()


# 수치 특성을 전처리하기 위한 파이프라인을 만듭니다(0.20 버전에 새로 추가된 `SimpleImputer` 클래스로 변경합니다):

# In[81]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[82]:


housing_num_tr


# ### future_encoders.py를 사용한 방법 ==========================
# 사이킷런의 0.20 버전에 포함될 `ColumnTransformer`를 사용하면 책의 예제에서처럼 `DataFrameSelector`와 `FeatureUnion`을 사용하지 않고 간단히 전체 파이프라인을 만들 수 있습니다. 아직 사이킷런 0.20 버전이 릴리스되기 전이므로 여기서는 future_encoders.py에 `ColumnTransformer`를 넣어 놓고 사용합니다.
# 
# 사이킷런 0.20 버전에 추가된 `sklearn.compose.ColumnTransformer`로 코드를 변경합니다.

# In[83]:


# from future_encoders import ColumnTransformer
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(categories='auto'), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared


# ### ====================================================

# 판단스 DataFrame 컬럼의 일부를 선택하는 변환기를 만듭니다:

# In[84]:


from sklearn.base import BaseEstimator, TransformerMixin

# 사이킷런이 DataFrame을 바로 사용하지 못하므로
# 수치형이나 범주형 컬럼을 선택하는 클래스를 만듭니다.
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


# 하나의 큰 파이프라인에 이들을 모두 결합하여 수치형과 범주형 특성을 전처리합니다:

# 0.20 버전에 추가된 `SimpleImputer`를 사용합니다.

# In[87]:


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', OneHotEncoder(categories='auto')),
    ])


# ### future_encoders.py를 사용한 방법 ==========================

# In[88]:


cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', OneHotEncoder(categories='auto', sparse=False)),
    ])


# ### ====================================================

# 사이킷런 0.20 버전에 추가된 `ColumnTransformer`로 만든 `full_pipline`을 사용합니다:

# In[89]:


# from sklearn.pipeline import FeatureUnion

# full_pipeline = FeatureUnion(transformer_list=[
#         ("num_pipeline", num_pipeline),
#         ("cat_pipeline", cat_pipeline),
#     ])
full_pipeline = ColumnTransformer([
        ("num_pipeline", num_pipeline, num_attribs),
        ("cat_encoder", OneHotEncoder(categories='auto'), cat_attribs),
    ])


# In[90]:


housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared


# In[91]:


housing_prepared.shape


# # 모델 선택과 훈련

# In[92]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# In[93]:


# 훈련 샘플 몇 개를 사용해 전체 파이프라인을 적용해 보겠습니다.
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("예측:", lin_reg.predict(some_data_prepared))


# 실제 값과 비교합니다:

# In[94]:


print("레이블:", list(some_labels))


# In[95]:


some_data_prepared


# In[96]:


from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[97]:


from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(housing_labels, housing_predictions)
lin_mae


# In[98]:


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)


# In[99]:


housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# # 모델 세부 튜닝

# In[100]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# In[101]:


def display_scores(scores):
    print("점수:", scores)
    print("평균:", scores.mean())
    print("표준편차:", scores.std())

display_scores(tree_rmse_scores)


# In[102]:


lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# 사이킷런 0.22 버전에서 랜덤 포레스트의 `n_estimator` 기본값이 10에서 100으로 변경됩니다. 0.20 버전에서 `n_estimator` 값을 지정하지 않을 경우 이에 대한 경고 메세지가 나옵니다. 경고 메세지를 피하기 위해 명시적으로 `n_estimator`를 10으로 설정합니다.

# In[103]:


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)


# In[104]:


housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[105]:


from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


# In[106]:


scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
pd.Series(np.sqrt(-scores)).describe()


# In[107]:


from sklearn.svm import SVR

svm_reg = SVR(kernel="linear")
svm_reg.fit(housing_prepared, housing_labels)
housing_predictions = svm_reg.predict(housing_prepared)
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)
svm_rmse


# In[108]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    # 하이퍼파라미터 12(=3×4)개의 조합을 시도합니다.
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # bootstrap은 False로 하고 6(=2×3)개의 조합을 시도합니다.
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# 다섯 폴드에서 훈련하면 총 (12+6)*5=90번의 훈련이 일어납니다.
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', 
                           return_train_score=True, n_jobs=-1)
grid_search.fit(housing_prepared, housing_labels)


# 최상의 파라미터 조합:

# In[109]:


grid_search.best_params_


# In[110]:


grid_search.best_estimator_


# 그리드서치에서 테스트한 하이퍼파라미터 조합의 점수를 확인합니다:

# In[111]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[112]:


pd.DataFrame(grid_search.cv_results_)


# In[113]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', 
                                random_state=42, n_jobs=-1)
rnd_search.fit(housing_prepared, housing_labels)


# In[114]:


cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[115]:


feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# 사이킷런 0.20 버전의 `ColumnTransformer`를 사용했기 때문에 `full_pipeline`에서 `cat_encoder`를 가져옵니다. 즉 `cat_pipeline`을 사용하지 않았습니다:

# In[116]:


extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
# cat_encoder = cat_pipeline.named_steps["cat_encoder"]
cat_encoder = full_pipeline.named_transformers_["cat_encoder"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


# In[117]:


final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[118]:


final_rmse


# 테스트 RMSE에 대한 95% 신뢰 구간을 계산할 수 있습니다:

# In[119]:


from scipy import stats


# In[120]:


confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
mean = squared_errors.mean()
m = len(squared_errors)

np.sqrt(stats.t.interval(confidence, m - 1,
                         loc=np.mean(squared_errors),
                         scale=stats.sem(squared_errors)))


# 다음과 같이 수동으로 계산할 수도 있습니다:

# In[121]:


tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)
tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - tmargin), np.sqrt(mean + tmargin)


# 또는 t 점수 대신 z 점수를 사용할 수도 있습니다:

# In[122]:


zscore = stats.norm.ppf((1 + confidence) / 2)
zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)


# # 추가 내용

# ## 전처리와 예측을 포함한 파이프라인

# In[123]:


full_pipeline_with_predictor = Pipeline([
        ("preparation", full_pipeline),
        ("linear", LinearRegression())
    ])

full_pipeline_with_predictor.fit(housing, housing_labels)
full_pipeline_with_predictor.predict(some_data)


# ## joblib을 사용한 모델 저장

# In[124]:


my_model = full_pipeline_with_predictor


# In[125]:


from sklearn.externals import joblib
joblib.dump(my_model, "my_model.pkl") # DIFF
#...
my_model_loaded = joblib.load("my_model.pkl") # DIFF

