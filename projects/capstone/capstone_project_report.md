
# Machine Learning Engineer Nanodegree
## Capstone Project
Ian Gregson
September 22nd 2018

# 1. Definition

### Project Overview

In this section, look to provide a high-level overview of the project in layman’s terms. Questions to ask yourself when writing this section:
- _Has an overview of the project been provided, such as the problem domain, project origin, and related datasets or input data?_
- _Has enough background information been given so that an uninformed reader would understand the problem domain and following problem statement?_


The domain background for this project is that it comes directly from my work for a startup that is building a customer data platform that helps users maximize their account based recurring revenue. In this field, BizOps profressional leverage large quantities of data from multiple sources such as CRMs or customer engagement analytics tools. The platform we build collates thiss data to provide a single view of an account (a non trivial problem in this space) but much of the insights that are gleaned from this large quantity of data come from the judgements of BizOps professionals studying the data.

<p align="center">
  <img width="460" height="300" src="images/model_cv_box_plot.png">
</p>

The project seeks to take steps into a new way of deriving insights from this large suite of structured data. By automating some aspects of the data analysis with machine learning models will free up the professionals to apply their extensive domain expertise in solving new problems.

I have a personal motivation in undertaking this project because I want to help push my team to become leaders in our field.

### Problem Statement

In this section, you will want to clearly define the problem that you are trying to solve, including the strategy (outline of tasks) you will use to achieve the desired solution. You should also thoroughly discuss what the intended solution will be for this problem. Questions to ask yourself when writing this section:
- _Is the problem statement clearly defined? Will the reader understand what you are expecting to solve?_
- _Have you thoroughly discussed how you will attempt to solve the problem?_
- _Is an anticipated solution clearly defined? Will the reader understand what results you are looking for?_


The problem to be solved in this project is the problem of opportunity scoring. There is historical data that shows which opportunities eventually convert to become accounts with annually recurring revenue. BizOps practitioners will use this historical data to make judgements on what active oppoortunities are most likely to convert by comparing their features with the features of examples from the historical data.

This project seeks to build a machine learning model that will make these judgements.

To build that model, this project will work through the following steps:

* Establish and prepare evaluation metrics
* Explore the dataset and establish necessary pre-processing steps
* Establish a benchmark using a dummy classifier
* Carry out pre-processing of the dataset
* Implement a number of models using grid search to establish the best hyperparameters
* Train and validate the models on the dataset
* Refine the model that looks to have the best performance
* Evaluate the performance of the model on the dataset against the benchmark and establish whether or not the model is viable for further use

The model will be deemed acceptable for furhter use if the predictions it makes are better than those returned by the dummy classifier.


### Metrics

In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in your project. These calculations and metrics should be justified based on the characteristics of the problem and problem domain. Questions to ask yourself when writing this section:
- _Are the metrics you’ve chosen to measure the performance of your models clearly discussed and defined?_
- _Have you provided reasonable justification for the metrics chosen based on the problem and solution?_


Since this project aims to solve a classification problem, I expect the F1 Score metric will provide suitable evaluation of the performance of the benchmark and solution models.

The F1 Score will provide a numeric means of judging the performance that can help more definiteively assess the performance of final model and ascertain whether or not it can be used further for opportunity scoring. The F1 Score metric should help give a good balance between _recall_ and _precision_ since it is unlikely the dataset will have an equal of number of samples in each of the two classes.

## 2. Analysis

### Data Exploration

In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
- _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
- _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_
- _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_
- _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_

**DataFrame Info**

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 33338 entries, 0 to 33337
Data columns (total 58 columns):
Account_Region__c                             25312 non-null object
Account_Theater__c                            10070 non-null object
ACV__c                                        33338 non-null float64
C_Contact_has_accepted_a_follow_on_step__c    33338 non-null bool
Consulting_Services_Amount__c                 33338 non-null float64
Created_by_Role__c                            33338 non-null object
Deal_Type__c                                  781 non-null object
Decision_Process__c                           0 non-null float64
Delivery_Type__c                              2 non-null object
Deployment_timeframe__c                       11 non-null object
Difference_between_Created_and_Modified__c    33338 non-null float64
DM_Close_Type__c                              33338 non-null object
DM_Opp_Age__c                                 33338 non-null float64
DM_Playbook_Stage__c                          22767 non-null object
DM_Playbook_Status__c                         33338 non-null object
Engagement_Mode__c                            18933 non-null object
Exchange_Reuse_Mgr__c                         33338 non-null bool
Follow_on_Meeting_Completed__c                33338 non-null bool
Follow_on_meeting_scheduled__c                33338 non-null bool
forecast__c                                   31570 non-null object
ForecastCategory                              33338 non-null object
HasOpportunityLineItem                        33338 non-null bool
Inbound__c                                    33338 non-null bool
Inbound_Source__c                             0 non-null float64
IsClosed                                      33338 non-null bool
IsSplit                                       33338 non-null bool
IsWon                                         33338 non-null bool
Key_Account__c                                33338 non-null bool
Large_Deal__c                                 33338 non-null bool
Lead_Passed_By_Group__c                       14124 non-null object
Lead_Passed_By_Name__c                        14823 non-null object
Lead_Passed_By_Role__c                        14803 non-null object
LeadSource                                    18209 non-null object
Lead_Source_Asset__c                          14519 non-null object
Lead_Source_Detail__c                         14832 non-null object
Lead_Type__c                                  33296 non-null object
Metric_Accept2Close__c                        33338 non-null float64
Metric_Create2Close__c                        33338 non-null float64
M_Is_decision_maker_mobilizer_champ__c        33338 non-null bool
N_Contact_has_bus_tech_goal_to_address__c     33338 non-null bool
New_and_Add_On_Subscription__c                27886 non-null float64
New_Business_Subscription__c                  14184 non-null float64
Number_of_Products__c                         33338 non-null float64
OA_Project_Prefix__c                          33338 non-null object
Opportunity_Classification__c                 18682 non-null object
Opportunity_Contact_Roles__c                  26576 non-null float64
Opportunity_Source__c                         32330 non-null object
Sales_Channel__c                              31384 non-null object
Services_Amount__c                            33338 non-null float64
Services_Attached__c                          33338 non-null bool
Stage__c                                      33338 non-null object
Subscr_Fields_Not_Populated__c                33338 non-null float64
Subscription_Amount__c                        33338 non-null float64
Total_List_Price__c                           33338 non-null float64
Who_is_leading_the_sale__c                    3554 non-null object
Amount                                        31845 non-null float64
StageName                                     33338 non-null object
Type                                          32928 non-null object
dtypes: bool(14), float64(17), object(27)
memory usage: 11.6+ MB
None
```

It appears the dataset contains a number of cells that are NaN or a data type that is unsuitable for using in an ML model. These will need to be removed during the project's data preprocessing step.

While this will reduce the number of features to an extent, once the remaining features are vectorized it is likely the total number will grow very large. To combat this, a feature selection step should be included during data preprocessing.

Of the 471 columns in the dataset, most are of the `object` data type. A number of these columns are most likely not useful and visual inspection of the data will be necessary to establish which of these categorical features need to be dropped from the dataset. The rest of the `boolean` and `object` type categorical features will then be vectorized before they go forward into feature selection.

There are a large number of columns with `null` values. If a feature has too many `null` values, it will not be useful to the final model and will be dropped. Otherwise, the `null` values will be filled using Pandas' `.fillna` method.

`StageName` is the label that this project is concerned with. As the graph shows, there are many different stage to which an opportunity can belong. This project is only concerned with two categories: 'Closed Won' and 'Closed Lost'. The dataset will be filtered to only include rows that have either of these labels.

It is also clear from the above graph that there are many more `Closed Won` samples than `Closed Lost`. This makes it all the more important that randomized test and validation slices are taken from the input data when training models. It will also be important to keep this in mind when assessing performance of the model: the uneven number of samples in each class anticipates a that a model could find _recall_ success but suffer from a high rate of false positives/negatives.

### TODO :: Exploratory Visualization

In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_


### Algorithms and Techniques

In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_


A number of algorithms will be employed in this project:

**Pandas**

* `.dropna()`
* `.fillna()`

These utilities are used to remove null samples from the dataset.

**Utilities**

* `make_scorer`
* `DummyClassifier`
* `DictVectorizer`
* `LabelEncoder`
* `MaxAbsScaler`
* `SelectKBest`
* `train_test_split`
* `Kfold`
* `cross_val_score`
* `GridSearchCV`

**Classifiers**

* `LogisticRegression`
* `GaussianNB`
* `KNeighborsClassifier`
* `SVC`
* `RandomForestClassifier`
* `lgb.sklearn.LGBMClassifier`
* `xgb.XGBClassifier`

Each classifier is employed at first with it's default configuration i.e. no parameters are passed to the constructor. Each of these models are tested with k-fold cross validation scored with the **f1 score** metric. The classifier that performs best in this pass will then be reconfigured with hyper-parameters using `GridSearchCV`.

### Benchmark

In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


For the purposes of this project, it is sufficient to build a model that is demonstrably better than a guess. Thus, it is sufficient to use sci-kit learn's dummy classifier to build a fake model. Then, the performance of this dummy model will be measured using the same metric (**f1 score**) as will be used to asses the final model.

The mean (**f1 score**) of the cross validated dummy classifier then gives a benchmark against which to judge the final model.

#### Metrics

#### Dummy Classifier

#### Benchmark

The DummyClassifier achieves an average of **f1 score** 73%. This establishes a baseline for assessing the success of the final modal. Given the dataset, a guess could be expected to be accurate in 73% of predictions. In order to demonstrate that the final model built in the project is better than a guess, it must a achieve a better average f1 score than 73%.

## 3. Methodolgy

### Data Preprocessing

In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_


A number of preprocessing steps will be required in order to ready the data for use in training a model. These steps are as follows:

* Load the raw dataset
* Filter the samples down to only those that include the labels that are useful for purposes here: `Closed won` and `Closed lost`
* Separate the labels column from the features and encode them
* Drop columns that have too many rows without a value and backfill the remain the columns
* Make a visual inspection of the remaining columns
* Drop a number of columns by name that are known to be of no use or are a direct proxy for the label column
* Make a list of the categorical features in the dataset and One Hot Encode them
* Write the processed dataset to CSV and make a final inspection

#### Load the raw dataset

#### Filter the samples down to only those that include the labels that are useful for purposes here: `Closed won` and `Closed lost`

#### Separate the labels column from the features and encode them

#### Drop columns that have too many rows without a value and backfill the remain the columns

#### Make a list of the categorical features in the dataset and One Hot Encode them

**Drop highly correlated features**

**Drop features that correlate with the labels**

#### Write the processed dataset to CSV

With the features encoded, the number of columns swells up to 6,625. This is likely too many and so a feature selection stage will be important during implementation.

### Implementation

In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_


#### Data loading

#### Train / Test Split

#### Scaling

#### Feature selection

#### Metrics

#### Model selection

<p align="center">
  <img width="460" height="300" src="images/model_cv_box_plot.png">
</p>

### Refinement

In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


#### Hyper Parameter Grid Search

## 4. Results

### Model Evaluation and Validation

In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_


### Jusitifcation

In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## 5. Conclusion

### Free-Form Visualization

In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_


### Reflection

In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_


### Improvement

In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_


--------------------------------------------------------------------------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?

