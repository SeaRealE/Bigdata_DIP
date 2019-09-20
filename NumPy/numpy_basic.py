#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc('figure',figsize=(10,6))
np.set_printoptions(precision=4, suppress=True)


# In[3]:


import numpy as np
my_arr = np.arange(1000000)      # numpy array
my_list = list(range(1000000))   # python list


# In[4]:


get_ipython().run_line_magic('time', 'for _ in range(10): my_arr2 = my_arr*2')


# In[5]:


get_ipython().run_line_magic('time', 'for _ in range(10): my_list2 = [x*2 for x in my_list]')


# In[6]:


import numpy as np
#Generate some random data
data = np.random.randn(2,3)
data


# In[7]:


data * 10


# In[8]:


data + data


# In[9]:


data.shape


# In[10]:


data.dtype


# In[11]:


data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
arr1


# In[12]:


data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)
arr2


# In[13]:


arr2.ndim


# In[14]:


arr2.shape


# In[15]:


arr1.dtype


# In[16]:


arr2.dtype


# In[17]:


np.zeros(10)


# In[18]:


np.zeros((3,6))


# In[19]:


np.ones(10)


# In[20]:


np.empty((2,3,2))


# In[21]:


np.arange(15)


# In[22]:


arr1 = np.array([1,2,3], dtype=np.float64)
arr2 = np.array([1,2,3], dtype=np.int32)
arr1.dtype


# In[23]:


arr2.dtype


# In[24]:


arr = np.array([1, 2, 3, 4, 5])
arr.dtype


# In[25]:


float_arr = arr.astype(np.float64)
float_arr.dtype


# In[26]:


arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
arr


# In[27]:


arr.astype(np.int32)


# In[28]:


numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)
numeric_strings.astype(float)


# In[29]:


int_array = np.arange(10)
calibers = np.array([.22, .270, .357, .380, .44, .50], dtype=np.float64)
int_array.astype(calibers.dtype)


# In[30]:


empty_unit32 = np.empty(8, dtype='u4')
empty_unit32


# In[31]:


arr = np.array([[1.,2.,3.],[4.,5.,6.]])
arr


# In[32]:


arr * arr


# In[33]:


arr - arr


# In[34]:


1/ arr


# In[35]:


arr ** 0.5


# In[36]:


arr2 = np.array([[0.,4.,1.],[7.,2.,12.]])
arr2


# In[37]:


arr2 > arr


# In[38]:


arr = np.arange(10)
arr


# In[39]:


arr[5]


# In[40]:


arr[5:8]


# In[41]:


arr[5:8] = 12
arr


# In[42]:


arr_slice = arr[5:8]
arr_slice


# In[43]:


arr_slice[1]=12345
arr


# In[44]:


arr_slice[:] = 64
arr


# In[45]:


arr2d = np.array([[1,2,3,],[4,5,6],[7,8,9]])
arr2d[2]


# In[46]:


arr2d[0][2]


# In[47]:


arr2d[0, 2]


# In[48]:


arr3d = np.array([[[1,2,3], [4,5,6]], [[7,8,9],[10,11,12]]])
arr3d


# In[49]:


arr3d[0]


# In[50]:


old_values = arr3d[0].copy()
arr3d[0]=42
arr3d


# In[51]:


arr3d[0] = old_values
arr3d


# In[52]:


arr3d[0] = old_values
arr3d


# In[53]:


arr3d[1, 0]


# In[54]:


x = arr3d[1]
x


# In[55]:


x[0]


# In[56]:


arr


# In[57]:


arr[5:8]=12
arr


# In[58]:


arr
arr[1:6]


# In[59]:


arr2d


# In[60]:


arr2d[:2]


# In[61]:


arr2d[:2, 1:]


# In[62]:


arr2d[2:, 1:]


# In[63]:


arr2d[2:, :1]


# In[64]:


arr2d[:,:1]


# In[65]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7,4)
names


# In[66]:


data


# In[67]:


names == 'Bob'


# In[68]:


data[names=='Bob']


# In[69]:


data[names == 'Bob', 2:]


# In[70]:


data[names == 'Bob', 3]


# In[71]:


names != 'Bob'


# In[72]:


data[~(names == 'Bob')]


# In[73]:


cond = names == 'Bob'
cond


# In[75]:


data[~cond]


# In[76]:


mask = (names == 'Bob') | (names == 'Will')
mask
data[mask]


# In[78]:


data[data < 0] = 0
data


# In[79]:


data[names != 'Joe'] = 7
data


# In[80]:


arr = np.empty((8,4))
for i in range(8):
    arr[i] = i
arr


# In[81]:


arr[[4,3,0,6]]


# In[82]:


arr[[-3,-5,-7]]


# In[83]:


arr = np.arange(32).reshape((8,4))
arr


# In[84]:


arr[[1, 5, 7, 2], [0,3,1,2]]


# In[85]:


arr


# In[89]:


arr[[1,5,7,2]][:,[0,3,1,2]]


# In[91]:


arr = np.arange(15).reshape((3, 5))
arr


# In[93]:


arr.T


# In[94]:


arr = np.random.randn(6, 3)
arr


# In[95]:


np.dot(arr.T, arr)


# In[96]:


arr = np.arange(16).reshape((2,2,4))
arr


# In[97]:


arr.transpose((1,0,2))


# In[98]:


arr


# In[99]:


arr.swapaxes(1,2)


# In[101]:


arr.swapaxes(0,2)


# In[102]:


arr.swapaxes(0,1)


# In[104]:


arr = np.arange(10)
arr


# In[106]:


np.sqrt(arr)


# In[107]:


np.exp(arr)


# In[108]:


x = np.random.randn(8)
y = np.random.randn(8)
x


# In[109]:


y


# In[110]:


np.maximum(x, y)


# In[111]:


arr = np.random.randn(7) * 5
arr


# In[116]:


remainder, whole_part = np.modf(arr)
remainder


# In[117]:


whole_part


# In[119]:


arr


# In[120]:


np.sqrt(arr)


# In[121]:


np.sqrt(arr, arr)


# In[122]:


arr


# In[123]:


points = np.arange(-5, 5, 0.01)
xs, ys = np.meshgrid(points, points)
ys


# In[124]:


z = np.sqrt(xs ** 2 + ys ** 2)
z


# In[125]:


import matplotlib.pyplot as plt
plt.imshow(z, cmap=plt.cm.gray); plt.colorbar()
plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")


# In[131]:


xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])


# In[135]:


result = [(x if c else y)
         for x, y, c in zip(xarr, yarr, cond)]
result


# In[133]:


result = np.where(cond, xarr, yarr)
result


# In[136]:


arr =np.random.randn(4, 4)
arr


# In[137]:


arr > 0


# In[138]:


np.where(arr > 0, 2, -2)


# In[139]:


np.where(arr > 0, 2, arr)


# In[ ]:





# In[ ]:




