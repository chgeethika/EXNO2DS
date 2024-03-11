# EXNO2DS
# AIM:
 To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('/content/titanic_dataset.csv')
df
```
![Screenshot 2024-03-11 213941](https://github.com/chgeethika/EXNO2DS/assets/142209368/d2d0888f-8872-41cb-bf38-26f313458faa)
```
df.info()
```
![Screenshot 2024-03-11 214157](https://github.com/chgeethika/EXNO2DS/assets/142209368/8e3a9136-c845-457e-a99c-114121a5862f)
```
df.shape
```
![Screenshot 2024-03-11 214254](https://github.com/chgeethika/EXNO2DS/assets/142209368/2f53beaf-5351-4226-8900-0d012ae6b70d)
```
df.head(4)
```
![Screenshot 2024-03-11 214553](https://github.com/chgeethika/EXNO2DS/assets/142209368/702bc350-5e76-442c-b63e-5ce47e937af6)

```
df.tail(2)
```
![Screenshot 2024-03-11 214606](https://github.com/chgeethika/EXNO2DS/assets/142209368/638d20d9-188a-414e-a192-2abdac52cbfb)
```
df.describe()
```
![Screenshot 2024-03-11 214830](https://github.com/chgeethika/EXNO2DS/assets/142209368/1f4f8974-ad11-4372-a250-957fd3ea4bb6)

```
df.set_index("PassengerId",inplace=True)
df.describe()
```
![Screenshot 2024-03-11 214840](https://github.com/chgeethika/EXNO2DS/assets/142209368/12df9709-ab69-4b67-a267-97069c112588)
```
df.nunique()
```
![Screenshot 2024-03-11 215045](https://github.com/chgeethika/EXNO2DS/assets/142209368/da89185c-e2e6-4ac0-a124-ba100743152b)

```
per=(df["Survived"].value_counts()/df.shape[0]*100).round(2)
per
```
![Screenshot 2024-03-11 215054](https://github.com/chgeethika/EXNO2DS/assets/142209368/0ded5218-9f4a-4fe2-ba7d-8f4031c1dd2a)

```
sns.countplot(data=df,x="Survived")
```
![Screenshot 2024-03-11 215123](https://github.com/chgeethika/EXNO2DS/assets/142209368/ff84c43d-7c1c-4111-b056-29ced4beaa30)

```
df
```
![Screenshot 2024-03-11 215206](https://github.com/chgeethika/EXNO2DS/assets/142209368/3c610713-e8f9-45be-bd40-58ec665aff3f)
```
df.Pclass.unique()
```
![Screenshot 2024-03-11 215436](https://github.com/chgeethika/EXNO2DS/assets/142209368/c2deec22-a79e-4630-972e-de77b0457964)

```
df.rename(columns= {'Sex':'Gender'}, inplace=True)
df
```
![Screenshot 2024-03-11 215457](https://github.com/chgeethika/EXNO2DS/assets/142209368/a6455839-491a-41e3-8735-24507c0c901c)

```
sns.catplot(x="Gender",col="Survived",kind="count",data=df,height=5,aspect=.7)
```
![Screenshot 2024-03-11 215522](https://github.com/chgeethika/EXNO2DS/assets/142209368/55bbd9c9-5d8b-47cc-93de-5968ac637d81)
```
sns.catplot(x='Survived',hue="Gender",data=df,kind="count")
```
![Screenshot 2024-03-11 215820](https://github.com/chgeethika/EXNO2DS/assets/142209368/9516cb79-8fde-45d5-82b2-5cced81a4044)

```
df.boxplot(column="Age",by="Survived")
```
![Screenshot 2024-03-11 215834](https://github.com/chgeethika/EXNO2DS/assets/142209368/f9c345c8-854b-4429-aa4d-c44f33e07f73)

```
sns.scatterplot(x=df["Age"],y=df["Fare"])
```
![Screenshot 2024-03-11 215851](https://github.com/chgeethika/EXNO2DS/assets/142209368/b32bcf9d-c080-4a07-9899-2766ab0a2252)
```
sns.jointplot(x="Age",y="Fare",data=df)
```

![Screenshot 2024-03-11 220118](https://github.com/chgeethika/EXNO2DS/assets/142209368/1fa17c04-d347-477c-85ae-46a7446f473c)

```
fig,ax1 = plt.subplots(figsize=(8,5))
pt=sns.boxplot(ax=ax1,x='Pclass',y='Age',hue='Gender',data=df)
```
![Screenshot 2024-03-11 220142](https://github.com/chgeethika/EXNO2DS/assets/142209368/6a463f52-7d33-431a-9a00-3058b26dfe73)

```
sns.catplot(data=df,col = "Survived",x = "Gender",hue="Pclass",kind = "count")
```

![Screenshot 2024-03-11 220159](https://github.com/chgeethika/EXNO2DS/assets/142209368/5ff1dc43-713f-4833-9346-ffd63eb116fa)

```
corr = df.corr()
sns.heatmap(corr,annot=True)
```
![Screenshot 2024-03-11 220420](https://github.com/chgeethika/EXNO2DS/assets/142209368/b4bfef30-9a49-4440-9afd-db6d4af92eb2)

```
sns.pairplot(df)
```

![Screenshot 2024-03-11 220545](https://github.com/chgeethika/EXNO2DS/assets/142209368/fb20aa4b-99a0-4ea9-89d0-6c9e34413225)

![Screenshot 2024-03-11 220604](https://github.com/chgeethika/EXNO2DS/assets/142209368/c4398eb2-54eb-494f-9717-dc799fe1ea60)

![Screenshot 2024-03-11 220627](https://github.com/chgeethika/EXNO2DS/assets/142209368/e0d44f23-ea43-43aa-bbaf-bf85d93936a2)


# RESULT
The Exploratory Data Analysis on the given data set is executed successfully.
       
