#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
plt.plot([10, 20, 30, 40])
plt.show()


# In[3]:


import matplotlib.pyplot as plt
plt.plot([1,2,3,4], [12,34,25,15])
plt.show()


# In[4]:


import matplotlib.pyplot as plt
plt.title('ploting') # 제목 넣기
plt.plot([10, 20, 30, 40])
plt.show()


# In[5]:


import matplotlib.pyplot as plt
plt.title('legend')
plt.plot([10, 20, 30, 40], label='asc')
plt.plot([40, 30, 20, 10], label='dsc')
plt.legend(loc=5)#범례표시
plt.show()


# In[6]:


import matplotlib.pyplot as plt
plt.title('color')
plt.plot([10, 20, 30, 40], color = 'skyblue', label='skyblue')
plt.plot([40, 30, 20, 10], color = 'pink', label='pink')
plt.legend(loc=5)
plt.show()


# In[9]:


import matplotlib.pyplot as plt
plt.title('linestyle')

# red dashed 그래프
plt.plot([10, 20, 30, 40], color = 'r', linestyle='--', label='dashed')
#plt.plot([10, 20, 30, 40] 'r--', label='dashed')

#green dotted 그래프
plt.plot([40, 30, 20, 10], color = 'b', ls = ':', label='dotted')
#plt.plot([40, 30, 20, 10,], 'b:', label='dotted')
plt.legend(loc=5)
plt.show()


# In[10]:


import matplotlib.pyplot as plt
plt.title('marker')
plt.plot([10, 20, 30, 40], 'r.', label='circle') # red 원형 마커
plt.plot([40, 30, 20, 10], 'b^', label='triangle up') # green 삼각형 마커
plt.legend(loc=5)
plt.show()


# In[11]:


import matplotlib.pyplot as plt
plt.title('marker')
plt.plot([10, 20, 30, 40], 'r.--', label='circle')
plt.plot([40, 30, 20, 10], 'b^-', label='triangle up')
plt.legend(loc=5)
plt.show()


# In[2]:


import matplotlib.pyplot as plt
plt.hist([1, 1, 3, 4, 5, 6, 6, 7, 8, 10])
plt.show()


# In[4]:


import matplotlib.pyplot as plt
import random

dice = []
for i in range(6):
    dice.append(random.randint(1, 6))
print(dice)
plt.hist(dice,bins=6)
plt.show()


# In[7]:


import matplotlib.pyplot as plt
import random
dice = []
for i in range(10000):
    dice.append(random.randint(1, 6))
plt.hist(dice, bins = 6)
plt.show()


# In[9]:


import matplotlib.pyplot as plt
import random

result = []
for i in range(13):
    result.append(random.randint(1, 1000))
print(sorted(result))

plt.boxplot(result)
plt.show()


# In[11]:


import numpy as np
result = np.array(result)
print("1/4: " + str(np.percentile(result, 25)))
print("2/4: " + str(np.percentile(result, 50)))
print("3/4: " + str(np.percentile(result, 75)))


# In[ ]:




