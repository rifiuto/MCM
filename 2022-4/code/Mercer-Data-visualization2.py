#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
hair_dryer_data=pd.read_excel('hair_dryer.xlsx')
microwave_data=pd.read_excel('microwave.xlsx')
pacifier_data=pd.read_excel('pacifier.xlsx')




print(len(hair_dryer_data))
print(len(microwave_data))
print(len(pacifier_data))




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




hair_stars,hair_stars_ratio=check_good_ratio(hair_dryer_data)
microwave_stars,microwave_stars_ratio=check_good_ratio(microwave_data)
pacifier_stars,pacifier_stars_ratio=check_good_ratio(pacifier_data)




print(hair_stars,hair_stars_ratio)




import seaborn as sns
sns.axes_style("darkgrid")
explode =[0,0,0,0.3,0]
labels =['one stars','two stars','three stars','four stars','five stars']
colors = ['salmon','tan','darkorange','skyblue','khaki']
patches,l_text,p_text=plt.pie(x=pacifier_stars_ratio,labels=labels,#        
        explode=explode,#    Python

colors=colors, #        

autopct='%.3f%%',#        ,  3   

pctdistance=0.4, #             

labeldistance=0.7,#          

startangle=180,#         

center=(4,4),#       (   X  Y    )

radius=3.8,#       (   X  Y    )

counterclock= False,#        ,False       

#wedgeprops= {'linewidth':10,'edgecolor':'lavender'}#            
)#          
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




def time_count(df):
    time_=[1,2,3,4,5,7,8,9,10,11,12]
    for row in df['review_date']:
        time_list=row.split('/')
        time=int(time_list[2]+time_list[0].zfill(2)+time_list[1].zfill(2))
        time_.append(time)
    return time_




from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image




def get_all_text(df):
    a=''
    for row in df['review_headline']:
        a+=str(row)
        a+=' '
    return a




a=get_all_text(microwave_data)




mask = np.array(Image.open("C:/Users/Administrator/Pictures/   .jpg"))




I,J,K=mask.shape
for i in range(I):
    for j in range(J):
        for k in range(K):
            if mask[i][j][k]>=246:
                mask[i][j][k]=255
print(mask)




wordcloud = WordCloud(
#    ,      ,                   
mask=mask,
#      ,     
background_color='#FFFFFF',
#           
scale=1,
#         ,         
#font_path="/usr/share/fonts/bb5828/     .otf"
).generate(a)
#    
image_produce = wordcloud.to_image()
#    
#wordcloud.to_file("new_wordcloud.jpg")
#    
image_produce.show()






