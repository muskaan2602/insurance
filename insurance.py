#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as py 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv("insurance.csv")
df #dataframe


# EDA

# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.head()


# In[7]:


df.describe() #only numeric values ka description aayega 


# In[8]:


df.isnull().sum() #we will check if we have null values or not, we will remove them in data cleaning


# In[9]:


df.columns


# In[10]:


numeric_columns = ['age',  'bmi', 'children', 'charges'] #we will extract these columns 
#we will use loop now 
for col in numeric_columns:
    plt.figure(figsize=(6,4)) #itne pixels ka we will visualize our data 
    sns.histplot(df[col]) #we will create a histogram
# charges right skewed data hai 


# In[11]:


sns.countplot(x = df["children"])


# In[12]:


sns.countplot(x = df['sex'])


# In[13]:


sns.countplot(x = df["smoker"])


# In[14]:


# now we will make boxplots 
for col in numeric_columns:
    plt.figure(figsize = (6,4))
    sns.boxplot(x= df[col])


# In[15]:


# ages mein koi outliers nahi haii 
#baki sbme hai 
# these are the steps for eda
# now we will se correlation and for doing it so we will make the correlation graph
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only = True), annot = True) # we will use heatmap and then send the correaltion data to heatmap  aur numeric values ka hi heatmap ban payega 


# data cleaning and preprocessing

# In[16]:


df_cleaned = df.copy() #maine ek naya variable banaya aur usme pura dataset copy kardiya 
df_cleaned.head()
df_cleaned.shape


# In[17]:


df_cleaned.drop_duplicates(inplace=True) 


# In[18]:


df_cleaned.shape


# In[19]:


df_cleaned.isnull().sum()


# In[20]:


df_cleaned.dtypes


# In[21]:


df_cleaned['sex'].value_counts()


# In[22]:


df_cleaned['sex'] = df_cleaned['sex'].map({"male":0,"female":1}) #we have converted our data into numeric data 


# In[23]:


df_cleaned.head()


# In[24]:


df_cleaned['smoker'].value_counts(
)


# In[25]:


df_cleaned['smoker']= df_cleaned['smoker'].map({"no":0 ,"yes":1})


# In[26]:


df_cleaned.head()


# In[27]:


df_cleaned.rename(columns = {
    "sex":"isfemale",
    "smoker":"is_smoker"
}, inplace= True) #inplace se humari cheezein actual mein kaam kar jati hain


# In[28]:


df_cleaned.head() #we used label encoding here 


# In[29]:


df_cleaned["region"].value_counts() #we will use one hot encoding


# In[30]:


df_cleaned = pd.get_dummies(df_cleaned,columns=["region"],drop_first=True) #get dummies chahiye hume region column ke toh we inserted that in get dummies 
#we can write multiple columns too, we use drop first warna region will also get considered for the name of columns 


# In[31]:


df_cleaned.head()


# In[32]:


df_cleaned= df_cleaned.astype(int)


# In[33]:


df_cleaned


# #feature engineering and extraction

# In[34]:


sns.histplot(df['bmi'])


# In[35]:


df_cleaned['bmi_category'] = pd.cut(
    df_cleaned['bmi'], 
    bins=[0,18.5,24.9,29.9,float("inf")],
    labels=["underweight","normal","overweight","obese"]
) #cut option in pandas library 


# In[36]:


df_cleaned


# In[37]:


df_cleaned = pd.get_dummies(df_cleaned,columns=["bmi_category"],drop_first=True)


# In[38]:


df_cleaned=df_cleaned.astype(int)


# In[39]:


df_cleaned


# In[40]:


#now we will scale 
df_cleaned.columns


# In[41]:


from sklearn.preprocessing import StandardScaler
cols =["age","bmi","children"]  #there are the 3 columns jisme hum standard deviation lagakar same scale par le aaayenge 
scaler = StandardScaler()

df_cleaned[cols]=scaler.fit_transform(df_cleaned[cols])


# In[42]:


df_cleaned.head()


# In[43]:


#hum charges ko nhi chhedenge kyunki its an output variable and hum output variable ko nhi chhedte 
#now we will do feature extraction
from scipy.stats import pearsonr

# ----------------------------------
# Pearson Correlation Calculation ki kiska correlation zyda hai jo ache se predict karne mein help karega about charges
# ----------------------------------

# List of features to check against target
selected_features = [
    'age', 'bmi', 'children', 'isfemale', 'is_smoker',
    'region_northwest', 'region_southeast', 'region_southwest',
    'bmi_category_normal', 'bmi_category_overweight', 'bmi_category_obese'
]
correlations = {
    feature: pearsonr(df_cleaned[feature], df_cleaned['charges'])[0]
    for feature in selected_features
}  #yahan ek dict ban jayegi 
correlation_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Pearson Correlation'])
correlation_df.sort_values(by='Pearson Correlation', ascending=False)


# In[44]:


# humara correlation humesha aayega +1 aur -1 ke bich mein, if my value is above o.5 or -0.5 then its highly correlated features so we will definately use them 
# we can conside 0.25 to 0.5, o.2 to 0.1 are negotiable if the accuracy is high when we use them then we use then if not then we dont use them. 

cat_features = [
    'isfemale', 'is_smoker',
    'region_northwest', 'region_southeast', 'region_southwest',
    'bmi_category_normal', 'bmi_category_overweight', 'bmi_category_obese'
] #these are all categorical features and we will now use kqitosis where we compare category to category 




# In[45]:


from scipy.stats import chi2_contingency
import pandas as pd

alpha = 0.05

df_cleaned['charges_bin'] = pd.qcut(df_cleaned['charges'], q=4, labels=False)
chi2_results = {}

for col in cat_features:
    contingency = pd.crosstab(df_cleaned[col], df_cleaned['charges_bin'])
    chi2_stat, p_val, _, _ = chi2_contingency(contingency)
    decision = 'Reject Null (Keep Feature)' if p_val < alpha else 'Accept Null (Drop Feature)'
    chi2_results[col] = {
        'chi2_statistic': chi2_stat,
        'p_value': p_val,
        'Decision': decision
    }

chi2_df = pd.DataFrame(chi2_results).T
chi2_df = chi2_df.sort_values(by='p_value')
chi2_df


# In[46]:


final_df = df_cleaned[['age', 'isfemale', 'bmi', 'children', 'is_smoker', 'charges','region_southeast','bmi_category_obese']]
final_df


# In[47]:


df


# In[51]:


from sklearn.model_selection import train_test_split
X= final_df.drop("charges",axis = 1)
Y= final_df["charges"]


# In[53]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=42)


# In[54]:


#WE WILL USE LINEAR REGRESSION
from sklearn.linear_model import LinearRegression 


# In[56]:


model= LinearRegression()
model.fit(X_train,y_train)


# In[57]:


# our model is created now we have to see how our model is working 
y_predict = model.predict(X_test)


# In[58]:


#NOW WE WILL COMPARE BOTH ARE MODEL AND USE R2 TO DO SO 
from sklearn.metrics import r2_score

r2 = r2_score(y_test,y_predict)
r2


# In[59]:


# we are getting 80% similarity in our scores 
n = X_test.shape[0]
p= X_test.shape[1]
adjusted_r2= 1-((1-r2)*(n-1)/(n-p-1))


# In[60]:


adjusted_r2


# In[ ]:


# our model is performing good 80 % is good, we can deploy our model 

