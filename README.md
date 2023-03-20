![image](https://user-images.githubusercontent.com/122312679/226447289-254db61c-b78e-4fbf-bca3-203bed749ea1.png)

# Titanic Survival Prediction
Authored by Jae Heon Kim

## Overview
This project is intended for practice and aimed at improving skills in developing machine learning classification models. It is not a real-world project with a specific business problem to solve, but rather a means of honing techniques and exploring different approaches to modeling. The focus is on gaining experience in data preparation, exploratory data analysis, feature engineering, model selection, hyperparameter tuning, and model evaluation. The goal is to gain a deeper understanding of the principles and best practices of machine learning, as well as to build a solid foundation for tackling more complex real-world problems in the future.

## Business Understanding
The Titanic dataset can be used to predict the survival of passengers on board the Titanic. By developing a machine learning classification model, we can gain insights into the factors that contribute to survival, and potentially identify individuals who may be at higher risk of not surviving in similar situations. This could be valuable for companies or organizations in the travel industry, such as cruise ship companies or airlines, who want to ensure the safety of their passengers in the event of an emergency.

## Data Understanding
### Data Introduction
The Titanic dataset is a popular machine learning dataset that provides information on the fate of passengers aboard the Titanic, including whether they survived or not, as well as demographic and trip-related information.

### Source
The dataset comes from seaborn library.

### Data Description
The dataset contains information on 891 passengers who were aboard the Titanic, of which 342 survived and 549 perished. Each row in the dataset represents a single passenger and the columns contain information about that passenger, including:

`survived`: Whether the passenger survived (1) or not (0)

`pclass`: The passenger's ticket class (1st, 2nd, or 3rd)

`sex`: The passenger's gender

`age`: The passenger's age in years

`sibsp`: The number of siblings/spouses aboard the Titanic

`parch`: The number of parents/children aboard the Titanic

`fare`: The fare paid by the passenger

`cabin`: The passenger's cabin number

`embarked`: The port where the passenger embarked (C = Cherbourg, Q = Queenstown, S = Southampton)

`deck`: The level on the Titanic where a passenger's cabin was located

`alone`: The "Alone" variable indicates whether a passenger was traveling alone or with family members on the Titanic. It is a binary variable that takes the value 1 if the passenger was traveling alone, and 0 if the passenger was traveling with one or more family members.

![image](https://user-images.githubusercontent.com/122312679/226448635-ee92edc0-77d6-4e3c-9744-9099f7a146ed.png)

This is the binary outcome that shows whether the person has survived, 1 representing survived, 0 for not. We can effortless assume death for everyone and be correct for nearly 61% of the times. We will make sure our models do much better than this, also considering that this random guess took no money and time.

## Data Preparation

- We start off by splitting training set and test set.
- We imputed missing values with mean and modes depending on type of variable and shape of its distribution.
- For `deck` we chose to create a new category because missing values nearly accounted for 2/3 of the entire dataset.
- In this stage, we define the pipelines so modeling process is quicker.

## Modeling
In this modeling phase, we build different classification models and search for one that has the highest accuracy score. The imbalance in binary outcome isn't that great so there needs no additional work on that. Our baseline model would be our effortless guess of death for everyone, which yields an accuracy score of 61%. Then we will build different classificatio models and compare their performances:
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Random Forest
- Extra Gradient Boosting
- Voting Classifier
- Stacking Classifier

All of these models will find its best hyperparameters through GridSearchCV. For information on detailed hyperparameters, please refer to the ipynb file. The graph below compares the results of our best-tuned models:

![image](https://user-images.githubusercontent.com/122312679/226452279-a041a0d8-4748-4d35-ba7d-28a4bf068c10.png)

We will select Stacking classifier as our final prediction model, which combined the best-tuned random forest, k-nearest neighbors and xg boost as estimators and used logistic regression as the final model. Its accuracy score of 83.6% on test data is the highest of all models.

## Evaluation
We will now evaluate what features in our final model are most indicative of the survival outcome.

![image](https://user-images.githubusercontent.com/122312679/226453847-735232e6-3724-4b98-aa9c-8dfa16cacb55.png)

