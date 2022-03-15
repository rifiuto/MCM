#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
hair_dryer_data=pd.read_excel('hair_dryer.xlsx')
microwave_data=pd.read_excel('microwave.xlsx')
pacifier_data=pd.read_excel('pacifier.xlsx')


# In[2]:


print(len(hair_dryer_data))
print(len(microwave_data))
print(len(pacifier_data))


# In[4]:


def check_good_ratio(df):
    one=0
    two=0
    three=0
    four=0
    five=0
    all_=len(df)
    for index,row in df.iterrows():
        if row['star_rating']==1:
            one+=1
        elif row['star_rating']==2:
            two+=1
        elif row['star_rating']==3:
            three+=1
        elif row['star_rating']==4:
            four+=1
        elif row['star_rating']==5:
            five+=1
    all_list=[one,two,three,four,five]
    all_list1=[a/all_ for a in all_list]
    return all_list,all_list1


# In[6]:


hair_stars,hair_stars_ratio=check_good_ratio(hair_dryer_data)
microwave_stars,microwave_stars_ratio=check_good_ratio(microwave_data)
pacifier_stars,pacifier_stars_ratio=check_good_ratio(pacifier_data)


# In[7]:


print(hair_stars,hair_stars_ratio)


# In[146]:


import seaborn as sns
sns.axes_style("darkgrid")
explode =[0,0,0,0.3,0]
labels =['one stars','two stars','three stars','four stars','five stars']
colors = ['salmon','tan','darkorange','skyblue','khaki']
patches,l_text,p_text=plt.pie(x=pacifier_stars_ratio,labels=labels,#添加编程语言标签
        explode=explode,#突出显示Python

colors=colors, #设置自定义填充色

autopct='%.3f%%',#设置百分比的格式,保留3位小数

pctdistance=0.4, #设置百分比标签和圆心的距离

labeldistance=0.7,#设置标签和圆心的距离

startangle=180,#设置饼图的初始角度

center=(4,4),#设置饼图的圆心(相当于X轴和Y轴的范围)

radius=3.8,#设置饼图的半径(相当于X轴和Y轴的范围)

counterclock= False,#是否为逆时针方向,False表示顺时针方向

#wedgeprops= {'linewidth':10,'edgecolor':'lavender'}#设置饼图内外边界的属性值
)#设置文本标签的属性值
for t in p_text:
    t.set_size(17)

for t in l_text:
    t.set_size(17)
plt.xticks(())

plt.yticks(())
plt.title('pacifier_stars_ratio',y=-1.18,fontsize=18)
plt.legend(patches,
           labels,
           fontsize=18,
           #title="star rank",
           loc="center left",
           bbox_to_anchor=(2, 0, 1, -1))

plt.show()


# In[ ]:


def time_count(df):
    time_=[1,2,3,4,5,7,8,9,10,11,12]
    for row in df['review_date']:
        time_list=row.split('/')
        time=int(time_list[2]+time_list[0].zfill(2)+time_list[1].zfill(2))
        time_.append(time)
    return time_


# In[67]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image


# In[44]:


def get_all_text(df):
    a=''
    for row in df['review_headline']:
        a+=str(row)
        a+=' '
    return a


# In[99]:


a=get_all_text(microwave_data)


# In[101]:


mask = np.array(Image.open("C:/Users/Administrator/Pictures/微波炉.jpg"))


# In[102]:


I,J,K=mask.shape
for i in range(I):
    for j in range(J):
        for k in range(K):
            if mask[i][j][k]>=246:
                mask[i][j][k]=255
print(mask)


# In[103]:


wordcloud = WordCloud(
# 遮罩层,除白色背景外,其余图层全部绘制（之前设置的宽高无效）
mask=mask,
#默认黑色背景,更改为白色
background_color='#FFFFFF',
#按照比例扩大或缩小画布
scale=1,
# 若想生成中文字体,需添加中文字体路径
#font_path="/usr/share/fonts/bb5828/逐浪雅宋体.otf"
).generate(a)
#返回对象
image_produce = wordcloud.to_image()
#保存图片
#wordcloud.to_file("new_wordcloud.jpg")
#显示图像
image_produce.show()


# In[ ]:




