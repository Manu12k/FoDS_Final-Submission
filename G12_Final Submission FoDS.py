

########################################
########## Import ######################
#######################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, auc, RocCurveDisplay, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
import openpyxl
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from scipy import interp
import seaborn as sns
import scipy.stats as stats




#load the data set with appropriate dtypes (data transformation)
###### MATCH THE PATH TO THE DATASET IN YOUR COMPUTER ######

##################################
########## MATHS Dataset ##########
##################################
df_maths = pd.read_csv("../data/Maths.csv",
                      dtype = {
                          "school": "category",
                          "sex": "category",
                          "age": np.int64,
                          "address": "category",
                          "famsize": "category",
                          "Pstatus": "category",
                          "Mjob": "category",
                          "Fjob": "category",
                          "reason": "category",
                          "guardian": "category",
                          "failures": np.int64,
                          "schoolsup": "category",
                          "famsup": "category",
                          "paid": "category",
                          "activities": "category",
                          "nursery": "category",
                          "higher": "category",
                          "internet": "category",
                          "romantic": "category",
                          "absences": np.int64,
                          "G1": np.int64,
                          "G2": np.int64,
                          "G3": np.int64
                      })

df_maths["Medu"] = df_maths["Medu"].astype(pd.CategoricalDtype(categories=[0, 1, 2, 3, 4], ordered=True))
df_maths["Fedu"] = df_maths["Fedu"].astype(pd.CategoricalDtype(categories=[0, 1, 2, 3, 4], ordered=True))
df_maths["traveltime"] = df_maths["traveltime"].astype(pd.CategoricalDtype(categories=[1, 2, 3, 4], ordered=True))
df_maths["studytime"] = df_maths["studytime"].astype(pd.CategoricalDtype(categories=[1, 2, 3, 4], ordered=True))
df_maths["famrel"] = df_maths["famrel"].astype(pd.CategoricalDtype(categories=[1, 2, 3, 4, 5], ordered=True))
df_maths["freetime"] = df_maths["freetime"].astype(pd.CategoricalDtype(categories=[1, 2, 3, 4, 5], ordered=True))
df_maths["goout"] = df_maths["goout"].astype(pd.CategoricalDtype(categories=[1, 2, 3, 4, 5], ordered=True))
df_maths["Dalc"] = df_maths["Dalc"].astype(pd.CategoricalDtype(categories=[1, 2, 3, 4, 5], ordered=True))
df_maths["Walc"] = df_maths["Walc"].astype(pd.CategoricalDtype(categories=[1, 2, 3, 4, 5], ordered=True))
df_maths["health"] = df_maths["health"].astype(pd.CategoricalDtype(categories=[1, 2, 3, 4, 5], ordered=True))

##################################
########## PORTU Dataset #########
##################################
df_portu = pd.read_csv("../data/Portuguese.csv",
                      dtype = {
                          "school": "category",
                          "sex": "category",
                          "age": np.int64,
                          "address": "category",
                          "famsize": "category",
                          "Pstatus": "category",
                          "Mjob": "category",
                          "Fjob": "category",
                          "reason": "category",
                          "guardian": "category",
                          "failures": np.int64,
                          "schoolsup": "category",
                          "famsup": "category",
                          "paid": "category",
                          "activities": "category",
                          "nursery": "category",
                          "higher": "category",
                          "internet": "category",
                          "romantic": "category",
                          "absences": np.int64,
                          "G1": np.int64,
                          "G2": np.int64,
                          "G3": np.int64
                      })

df_portu["Medu"] = df_portu["Medu"].astype(pd.CategoricalDtype(categories=[0, 1, 2, 3, 4], ordered=True))
df_portu["Fedu"] = df_portu["Fedu"].astype(pd.CategoricalDtype(categories=[0, 1, 2, 3, 4], ordered=True))
df_portu["traveltime"] = df_portu["traveltime"].astype(pd.CategoricalDtype(categories=[1, 2, 3, 4], ordered=True))
df_portu["studytime"] = df_portu["studytime"].astype(pd.CategoricalDtype(categories=[1, 2, 3, 4], ordered=True))
df_portu["famrel"] = df_portu["famrel"].astype(pd.CategoricalDtype(categories=[1, 2, 3, 4, 5], ordered=True))
df_portu["freetime"] = df_portu["freetime"].astype(pd.CategoricalDtype(categories=[1, 2, 3, 4, 5], ordered=True))
df_portu["goout"] = df_portu["goout"].astype(pd.CategoricalDtype(categories=[1, 2, 3, 4, 5], ordered=True))
df_portu["Dalc"] = df_portu["Dalc"].astype(pd.CategoricalDtype(categories=[1, 2, 3, 4, 5], ordered=True))
df_portu["Walc"] = df_portu["Walc"].astype(pd.CategoricalDtype(categories=[1, 2, 3, 4, 5], ordered=True))
df_portu["health"] = df_portu["health"].astype(pd.CategoricalDtype(categories=[1, 2, 3, 4, 5], ordered=True))


#check the dataset maths
df_maths


#check the dataset portu
df_portu


#check whether dtypes are correct
df_maths.dtypes


#check whether dtypes are correct
df_portu.dtypes



##################################
######## Passes and Fails ########
##################################

#feature engineering - creating a new variable Ghigh, which is 1 if G3 >= 10 and 0 otherwise.
#This is the class variable of our classifiers
df_maths['Ghigh'] = (df_maths['G3'] >= 10).astype(int)
df_maths['Ghigh'] = df_maths["Ghigh"].astype(pd.CategoricalDtype(categories = [0, 1], ordered = False))
df_portu['Ghigh'] = (df_portu['G3'] >= 10).astype(int)
df_portu['Ghigh'] = df_portu["Ghigh"].astype(pd.CategoricalDtype(categories = [0, 1], ordered = False))



##################################
######### Data Overview ##########
##################################

#Define a function to get an overview of each dataset
def df_overview(df, target_classes, subjects):
    print('-'*40)
    print('-'*13, '{} DATASET'.format(subjects), '-'*12)
    print('-'*40)
    print('No. of students in the dataset:', ' '*4, '{}'.format(df.shape[0]))
    print('No. of attributes in the dataset:', ' '*3, '{}'.format(df.shape[1]-1))
    print('No. of duplicated rows in the dataset: {}'.format(df.duplicated().sum(axis=0)))
    print('No. of missing values in the dataset:  {}'.format(df.isna().sum(axis=0).sum()))
    #checking whether the class variable (Ghigh) is balanced
    for i in range(2):
        print('No. of {}:'.format(target_classes[i]), ' '*23, df["Ghigh"].value_counts()[i])
        print('Proportion of {}:'.format(target_classes[i]), ' '*13, '{:.4f}'.format(df["Ghigh"].value_counts()[i] / df.shape[0]))
    if (df["Ghigh"].value_counts()[1] / df.shape[0]) > 0.6 or (df["Ghigh"].value_counts()[1] / df.shape[0]) < 0.4:
        print('The class distribution is not balanced.')
    else:
        print('The class distribution is balanced.')
    print('-'*40)

#Create lists for dfs, target classes and subjects
dfs = [df_maths, df_portu]
pass_fail = ["fail", "pass"]
maths_portu = ["MATHS", "PORTU"]

#Output the data overview
for i in range(2):
    df_overview(dfs[i], pass_fail, maths_portu[i])
    print()


##################################
#### Normality and Comparison ####
##################################

# Normality Maths
for column in df_maths.columns:
    if pd.api.types.is_numeric_dtype(df_maths[column]):
        shapiro_stat, shapiro_p = stats.shapiro(df_maths[column])
        if shapiro_stat > 0.9 and shapiro_p < 0.05:
            print(f'Shapiro-Wilk Test of {column}: Statistic={shapiro_stat}, p-value={shapiro_p}')

# Normality Portu
for column in df_portu.columns:
    if pd.api.types.is_numeric_dtype(df_portu[column]):
        shapiro_stat, shapiro_p = stats.shapiro(df_portu[column])
        if shapiro_stat > 0.9 and shapiro_p < 0.05:
            print(f'Shapiro-Wilk Test of {column}: Statistic={shapiro_stat}, p-value={shapiro_p}')

# Age
passes_maths = df_maths[df_maths['Ghigh'] == 1]['age']
fails_maths = df_maths[df_maths['Ghigh'] == 0]['age']
t_stat, p_value = stats.ttest_ind(passes_maths, fails_maths)
print(f'Maths Age T-Test: t_stat={t_stat}, p_value={p_value}')

passes_portu = df_portu[df_portu['Ghigh'] == 1]['age']
fails_portu = df_portu[df_portu['Ghigh'] == 0]['age']
t_stat, p_value = stats.ttest_ind(passes_portu, fails_portu)
print(f'Portu Age T-Test: t_stat={t_stat}, p_value={p_value}')

# Dalc
contingency_table = pd.crosstab(df_maths['Ghigh'], df_maths['Dalc'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f'Maths Dalc Chi-Square Test: chi2={chi2}, p-value={p}')

contingency_table = pd.crosstab(df_portu['Ghigh'], df_portu['Dalc'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f'Portu Dalc Chi-Square Test: chi2={chi2}, p-value={p}')

# Walc
contingency_table = pd.crosstab(df_maths['Ghigh'], df_maths['Walc'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f'Maths Walc Chi-Square Test: chi2={chi2}, p-value={p}')

contingency_table = pd.crosstab(df_portu['Ghigh'], df_portu['Walc'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f'Portu Walc Chi-Square Test: chi2={chi2}, p-value={p}')

# Sex
contingency_table = pd.crosstab(df_maths['Ghigh'], df_maths['sex'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f'Maths Sex Chi-Square Test: chi2={chi2}, p-value={p}')

contingency_table = pd.crosstab(df_portu['Ghigh'], df_portu['sex'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f'Portu Sex Chi-Square Test: chi2={chi2}, p-value={p}')

##################################
######### Visualisation ##########
##################################

fig = plt.figure(figsize = (12, 7), constrained_layout = True)
subfigs = fig.subfigures(2, 1)
subfigs[0].suptitle('MATHS dataset')
ax_1 = subfigs[0].subplots(1, 4)

# SEX MATH
cross_tab = pd.crosstab(df_maths['Ghigh'], df_maths['sex'])
cross_tab.plot(kind='bar', stacked=False, ax=ax_1[0])
ax_1[0].set_title('Pass/Fail subdivided by sex', size = 9)
ax_1[0].set_xlabel('Pass/Fail [1, 0]', size = 8)
ax_1[0].set_ylabel('Count', size = 8)
ax_1[0].legend(title = 'Sex', title_fontsize = 8, fontsize = 8)
ax_1[0].tick_params(axis = 'x', rotation = 0, labelsize = 7)
ax_1[0].tick_params(axis = 'y', labelsize = 7)

# DALC MATH
grouped_data_dalc = df_maths.groupby(['Dalc', 'Ghigh'], observed = True).size().unstack()
grouped_data_dalc.plot(kind='bar', ax=ax_1[1])
ax_1[1].set_title('Workday EtOH consumption\n subdivided in Pass/Fail', size = 9)
ax_1[1].set_xlabel('Dalc (1 - very low, 5 - very high)', size = 8)
ax_1[1].set_ylabel('Count', size = 8)
ax_1[1].legend(title = 'Outcome', labels = ['Fail', 'Pass'], title_fontsize = 8, fontsize = 8)
ax_1[1].tick_params(axis = 'x', rotation = 0, labelsize = 7)
ax_1[1].tick_params(axis = 'y', labelsize = 7)

# WALC MATH
grouped_data_walc = df_maths.groupby(['Walc', 'Ghigh'], observed = True).size().unstack()
grouped_data_walc.plot(kind = 'bar', ax = ax_1[2])
ax_1[2].set_title('Weekend EtOH consumption\n subdivided in Pass/Fail', size = 9)
ax_1[2].set_xlabel('Walc (1 - very low, 5 - very high)', size = 8)
ax_1[2].set_ylabel('Count', size = 8)
ax_1[2].legend(title = 'Outcome', labels = ['Fail', 'Pass'], title_fontsize = 8, fontsize = 8)
ax_1[2].tick_params(axis = 'x', rotation = 0, labelsize = 7)
ax_1[2].tick_params(axis = 'y', labelsize = 7)

# AGE MATH
sns.boxplot(x = 'age', y = 'Ghigh', data = df_maths, ax = ax_1[3], color = 'red')
ax_1[3].set_title('Pass/Fail age distribution', size = 9)
ax_1[3].set_xlabel('Age [years]', size = 8)
ax_1[3].set_ylabel('Pass/Fail [1, 0]', size = 8)
ax_1[3].tick_params(axis = 'x', rotation = 0, labelsize = 7)
ax_1[3].tick_params(axis = 'y', labelsize = 7)

subfigs[1].suptitle('PORTUGUESE dataset')
ax_2 = subfigs[1].subplots(1, 4)
# SEX PORT
cross_tab = pd.crosstab(df_portu['Ghigh'], df_portu['sex'])
cross_tab.plot(kind='bar', stacked=False, ax=ax_2[0])
ax_2[0].set_title('Pass/Fail subdivided by sex', size = 9)
ax_2[0].set_xlabel('Pass/Fail [1, 0]', size = 8)
ax_2[0].set_ylabel('Count', size = 8)
ax_2[0].legend(title = 'Sex', title_fontsize = 8, fontsize = 8, loc = 2)
ax_2[0].tick_params(axis = 'x', rotation = 0, labelsize = 7)
ax_2[0].tick_params(axis = 'y', labelsize = 7)

# DALC PORT
grouped_data_dalc = df_portu.groupby(['Dalc', 'Ghigh'], observed = True).size().unstack()
grouped_data_dalc.plot(kind='bar', ax=ax_2[1])
ax_2[1].set_title('Workday EtOH consumption\n subdivided in Pass/Fail', size = 9)
ax_2[1].set_xlabel('Dalc (1 - very low, 5 - very high)', size = 8)
ax_2[1].set_ylabel('Count', size = 8)
ax_2[1].legend(title = 'Outcome', labels = ['Fail', 'Pass'], title_fontsize = 8, fontsize = 8)
ax_2[1].tick_params(axis = 'x', rotation = 0, labelsize = 7)
ax_2[1].tick_params(axis = 'y', labelsize = 7)

# WALC PORT
grouped_data_walc = df_portu.groupby(['Walc', 'Ghigh'], observed = True).size().unstack()
grouped_data_walc.plot(kind = 'bar', ax = ax_2[2])
ax_2[2].set_title('Weekend EtOH consumption\n subdivided in Pass/Fail', size = 9)
ax_2[2].set_xlabel('Walc (1 - very low, 5 - very high)', size = 8)
ax_2[2].set_ylabel('Count', size = 8)
ax_2[2].legend(title = 'Outcome', labels = ['Fail', 'Pass'], title_fontsize = 8, fontsize = 8)
ax_2[2].tick_params(axis = 'x', rotation = 0, labelsize = 7)
ax_2[2].tick_params(axis = 'y', labelsize = 7)

# AGE PORT
sns.boxplot(x = 'age', y = 'Ghigh', data = df_portu, ax = ax_2[3], color = 'red')
ax_2[3].set_title('Pass/Fail age distribution', size = 9)
ax_2[3].set_xlabel('Age [years]', size = 8)
ax_2[3].set_ylabel('Pass/Fail [1, 0]', size = 8)
ax_2[3].tick_params(axis = 'x', rotation = 0, labelsize = 7)
ax_2[3].tick_params(axis = 'y', labelsize = 7)

plt.savefig('../output/visualisation1.png', dpi = 600)
plt.show()


##################################
########### Variables ############
##################################

# Nominal categorical variables
nominal_vars = ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian",
                "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]

# Ordinal categorical variables
ordinal_vars = ["Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout",
                "Dalc", "Walc", "health"]

# Numerical variables
numerical_vars = ['age', 'failures', 'absences']



##################################
######## One-Hot Encoding ########
##################################

# Make a copy of the df, just in case
df_maths_copy = df_maths.copy()
df_portu_copy = df_portu.copy()

# One-hot encoding for nominal categorical variables before train test split
# Maths data
df_math_encoded = pd.get_dummies(df_maths_copy, columns = nominal_vars, drop_first = True, dtype = int)

# Portuguese data
df_portu_encoded = pd.get_dummies(df_portu_copy, columns = nominal_vars, drop_first = True, dtype = int)




##################################
########### Splitting ############
##################################

# Maths df
X_maths = df_math_encoded.drop(["G1", "G2", "G3", "Ghigh"], axis=1)
y_maths = df_math_encoded["Ghigh"]

X_maths_train, X_maths_test, y_maths_train, y_maths_test = train_test_split(X_maths, y_maths, test_size = 0.2,
                                                                        stratify = y_maths, random_state = 42)

# Portuguese df
X_portu = df_portu_encoded.drop(["G1", "G2", "G3", "Ghigh"], axis=1)
y_portu = df_portu_encoded["Ghigh"]

X_portu_train, X_portu_test, y_portu_train, y_portu_test = train_test_split(X_portu, y_portu, test_size = 0.2,
                                                                            stratify = y_portu, random_state = 42)



##################################
############ Scaling #############
##################################

# Create copies of the data to avoid inplace operations
X_maths_train_scaled, X_maths_test_scaled = X_maths_train.copy(), X_maths_test.copy()
X_portu_train_scaled, X_portu_test_scaled = X_portu_train.copy(), X_portu_test.copy()

# Scaling was done seperately on the two datasets because distribution characteristics might be different
scaler_maths = StandardScaler()
X_maths_train_scaled[numerical_vars] = scaler_maths.fit_transform(X_maths_train_scaled[numerical_vars])
X_maths_test_scaled[numerical_vars] = scaler_maths.transform(X_maths_test_scaled[numerical_vars])

scaler_portu = StandardScaler()
X_portu_train_scaled[numerical_vars] = scaler_portu.fit_transform(X_portu_train_scaled[numerical_vars])
X_portu_test_scaled[numerical_vars] = scaler_portu.transform(X_portu_test_scaled[numerical_vars])

# Feature selection using Recursive Feature Elimination with Cross-Validation (RFECV)
######################################
########### RFECV 10 folds ###########
######################################

# Set with the intention of making interpretation easy
min_features_to_select = 7

# Define the F1 score as the scoring metric for RFECV. Chose F1 score because the class variable is imbalanced.
scorer = make_scorer(f1_score)


# Function to perform RFECV with L2 regularization
def rfecv_with_regularization(X_train_scaled, y_train, subject, min_features_to_select, scorer):
    # Set the no. of folds
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Hyperparameter tuning to get the best value for regularization strength
    param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
    logreg = LogisticRegression(class_weight='balanced', max_iter=1000, penalty='l2')
    grid_search = GridSearchCV(logreg, param_grid, cv=skf, scoring=scorer)  # to obtain the best value for C
    grid_search.fit(X_train_scaled, y_train)
    best_C = grid_search.best_params_['C']
    print("Best parameter for regularization strength (C) in", subject, ": ", best_C)

    # Feature selection with RFECV
    logreg_best = LogisticRegression(class_weight='balanced', max_iter=1000, penalty='l2', C=best_C)
    rfecv = RFECV(estimator=logreg_best, cv=skf, scoring=scorer, min_features_to_select=min_features_to_select)
    rfecv.fit(X_train_scaled, y_train)

    return rfecv, best_C


# Apply RFECV with regularization to both datasets
rfecv_maths, best_C_maths = rfecv_with_regularization(X_maths_train_scaled, y_maths_train, maths_portu[0],
                                                      min_features_to_select, scorer)
rfecv_portu, best_C_portu = rfecv_with_regularization(X_portu_train_scaled, y_portu_train, maths_portu[1],
                                                      min_features_to_select, scorer)

# Dashed line to improve visibility
print("-" * 115)

# Output the no. of optimal features of the math dataset
print("Optimal no. of features in the maths dataset: {}".format(rfecv_maths.n_features_))

# Output the chosen features of the maths dataset
chosen_features_maths = X_maths_train_scaled.columns[rfecv_maths.support_]
print("\nChosen features in the maths dataset: {}".format(chosen_features_maths))

# Dashed line to improve visibility
print("-" * 115)

# Output the no. of optimal features of the portuguese dataset
print("Optimal no. of features in the portuguese dataset: {}".format(rfecv_portu.n_features_))

# Output the chosen features of the portuguese dataset
chosen_features_portu = X_portu_train_scaled.columns[rfecv_portu.support_]
print("\nChosen features in the portuguese dataset: {}".format(chosen_features_portu))

# Dashed line to improve visibility
print("-" * 115)

# Plot number of features vs. cross-validation scores
n_scores_maths = len(rfecv_maths.cv_results_["mean_test_score"])
n_scores_portu = len(rfecv_portu.cv_results_["mean_test_score"])

# Create subplots
fig, ax = plt.subplots(2, 1, figsize=(8, 5))

# Subplot for the math dataset
ax[0].set_xlabel("Number of features selected")
ax[0].set_ylabel("Cross-validation score \n(F1 Score)")
ax[0].errorbar(range(min_features_to_select, n_scores_maths + min_features_to_select),
               rfecv_maths.cv_results_["mean_test_score"],
               yerr=rfecv_maths.cv_results_["std_test_score"], )
ax[0].set_title("Maths dataset")

# Subplot for the portuguese dataset
ax[1].set_xlabel("Number of features selected")
ax[1].set_ylabel("Cross-validation score \n(F1 Score)")
ax[1].errorbar(range(min_features_to_select, n_scores_portu + min_features_to_select),
               rfecv_portu.cv_results_["mean_test_score"],
               yerr=rfecv_portu.cv_results_["std_test_score"], )
ax[1].set_title("Portuguese dataset")

# Set a global title
fig.suptitle("Recursive Feature Elimination with \nCross-validation", x=0.56)
plt.tight_layout()
plt.show()

# Dashed line to improve visibility
print("-"*115)

####################################
####################################
############ Models ################
####################################
####################################


########################
####### LR Model #######
########################

# Function to perform hyperparameter tuning and cross-validation on the selected features
def cross_val_with_hyperparameter_tuning(X_train_scaled, y_train, chosen_features, subject, scorer):
    # Chose the optimal feature obtained from RFECV before
    X_train_selected = X_train_scaled.iloc[:, chosen_features]

    # Set the no. of folds
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Hyperparameter tuning to get the best value for regularization strength
    param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
    logreg = LogisticRegression(class_weight='balanced', max_iter=1000, penalty='l2')
    grid_search = GridSearchCV(logreg, param_grid, cv=skf, scoring=scorer)
    grid_search.fit(X_train_selected, y_train)
    best_C = grid_search.best_params_['C']
    print("Best parameter for regularization strength (C) in", subject, ": ", best_C)

    # Empty arrays to store evaluation metrics of each validation step
    accuracies = []
    recalls = []
    precisions = []
    f1s = []
    roc_aucs = []

    # Splitting
    for train_index, test_index in skf.split(X_train_selected, y_train):
        X_train_fold, X_test_fold = X_train_selected.iloc[train_index, :], X_train_selected.iloc[test_index, :]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        # Initialize the model
        model = LogisticRegression(class_weight="balanced", max_iter=1000, penalty='l2', C=best_C)
        # Fit to the model
        model.fit(X_train_fold, y_train_fold)

        # Test on the test fold in CV
        y_pred = model.predict(X_test_fold)
        y_pred_prob = model.predict_proba(X_test_fold)[:, 1]  # for the ROC AUC metric

        # Fill up the arrays for metrics
        accuracies.append(accuracy_score(y_test_fold, y_pred))
        recalls.append(recall_score(y_test_fold, y_pred))
        precisions.append(precision_score(y_test_fold, y_pred))
        f1s.append(f1_score(y_test_fold, y_pred))
        roc_aucs.append(roc_auc_score(y_test_fold, y_pred_prob))

    # Output the performance in training (= 10 fold CV)
    print("\nTraining set performance:")
    print("\nAccuracy: {:.4f} ± {:.4f}".format(np.mean(accuracies), np.std(accuracies)))
    print("Recall: {:.4f} ± {:.4f}".format(np.mean(recalls), np.std(recalls)))
    print("Precision: {:.4f} ± {:.4f}".format(np.mean(precisions), np.std(precisions)))
    print("F1 Score: {:.4f} ± {:.4f}".format(np.mean(f1s), np.std(f1s)))
    print("ROC AUC: {:.4f} ± {:.4f}".format(np.mean(roc_aucs), np.std(roc_aucs)))

    return model


# Convert chosen feature indices to numpy arrays for indexing
chosen_features_maths_indices = np.where(rfecv_maths.support_)[0]
chosen_features_portu_indices = np.where(rfecv_portu.support_)[0]

# Train models and evaluate on test sets
print("\nMaths dataset:")

# Train the LR model on the Math dataset
model_maths = cross_val_with_hyperparameter_tuning(X_maths_train_scaled, y_maths_train, chosen_features_maths_indices,
                                                   maths_portu[0], scorer)

# Extract the chosen optimal features from the TEST set
X_maths_test_selected = X_maths_test_scaled.iloc[:, chosen_features_maths_indices]

# Evaluate on the TEST set
y_maths_pred = model_maths.predict(X_maths_test_selected)
y_maths_pred_prob = model_maths.predict_proba(X_maths_test_selected)[:, 1]

# Output the performance on the TEST set of the Math dataset
print("\nTest set performance:")
print("\nAccuracy: {:.4f}".format(accuracy_score(y_maths_test, y_maths_pred)))
print("Recall: {:.4f}".format(recall_score(y_maths_test, y_maths_pred)))
print("Precision: {:.4f}".format(precision_score(y_maths_test, y_maths_pred)))
print("F1 Score: {:.4f}".format(f1_score(y_maths_test, y_maths_pred)))
print("ROC AUC: {:.4f}".format(roc_auc_score(y_maths_test, y_maths_pred_prob)))

# Dashed line to improve visibility
print("-" * 115)

print("\nPortuguese dataset:")

# Train the LR model on the Portuguese dataset
model_portu = cross_val_with_hyperparameter_tuning(X_portu_train_scaled, y_portu_train, chosen_features_portu_indices,
                                                   maths_portu[1], scorer)

# Extract the chosen optimal features from the TEST set
X_portu_test_selected = X_portu_test_scaled.iloc[:, chosen_features_portu_indices]

# Evaluate on the TEST set
y_portu_pred = model_portu.predict(X_portu_test_selected)
y_portu_pred_prob = model_portu.predict_proba(X_portu_test_selected)[:, 1]

# Output the performance on the TEST set of the Portuguese dataset
print("\nTest set performance:")
print("\nAccuracy: {:.4f}".format(accuracy_score(y_portu_test, y_portu_pred)))
print("Recall: {:.4f}".format(recall_score(y_portu_test, y_portu_pred)))
print("Precision: {:.4f}".format(precision_score(y_portu_test, y_portu_pred)))
print("F1 Score: {:.4f}".format(f1_score(y_portu_test, y_portu_pred)))
print("ROC AUC: {:.4f}".format(roc_auc_score(y_portu_test, y_portu_pred_prob)))


# # Analyzing feature importance

# # Coefficient extraction and visualisation

# A function to output coefficients of features in the LR models as a table
def output_coefficients_table(features, coefficients):
    # Create a DataFrame to store the coefficients
    coefficients_df = pd.DataFrame({'Feature': features, 'Coefficient': coefficients})
    # Format coefficient to 4 decimal places
    coefficients_df['Coefficient'] = coefficients_df['Coefficient'].apply(lambda x: "{:.4f}".format(x))
    # Print the DataFrame
    print(coefficients_df)


# Extract the coefficients of features in the LR models
coefficients_maths = model_maths.coef_[0]
coefficients_portu = model_portu.coef_[0]

print("Coefficients for the Maths dataset:")
print()
output_coefficients_table(chosen_features_maths, coefficients_maths)
print()

# Dashed line to improve visibility
print("-" * 115)

print("\nCoefficients for the Portuguese dataset:")
print()
output_coefficients_table(chosen_features_portu, coefficients_portu)

# Visualize coefficients for the Maths dataset
plt.figure(figsize=(10, 6))
plt.barh(chosen_features_maths, coefficients_maths)
plt.xlabel('Coefficient value')
plt.ylabel('Feature')
plt.title('Logistic Regression Coefficients - Maths dataset')
plt.tight_layout()
plt.savefig('../output/coeff_MATHS_LR.png')
plt.show()

# Visualize coefficients for the Portuguese dataset
plt.figure(figsize=(10, 6))
plt.barh(chosen_features_portu, coefficients_portu)
plt.xlabel('Coefficient value')
plt.ylabel('Feature')
plt.title('Logistic Regression Coefficients - Portuguese dataset')
plt.tight_layout()
plt.savefig('../output/coeff_PORTU_LR.png')
plt.show()


# Function to evaluate Models performance for DT and RF
#################################################
########## Performance evaluation ###############
#################################################
def eval_Performance(y_eval, X_eval, clf):
    # Make predictions
    y_pred = clf.predict(X_eval)
    y_pred_proba = clf.predict_proba(X_eval)[:, 1]

    # Evaluation
    precision = precision_score(y_eval, y_pred)
    recall = recall_score(y_eval, y_pred)
    f1 = f1_score(y_eval, y_pred)
    fp_rates, tp_rates, _ = roc_curve(y_eval, y_pred_proba)

    # Calculate the area under the roc curve using a sklearn function
    roc_auc = auc(fp_rates, tp_rates)

    return precision, recall, f1, roc_auc, fp_rates, tp_rates


# Function to combine mean and std into a single string for each metric, rounded to 4 decimal places, and update column labels for DT and RF
#################################
######## mean and std ###########
#################################
def combine_mean_std(mean_df, std_df):
    combined_df = mean_df.copy()
    new_columns = {}
    for col in mean_df.columns:
        mean_rounded = mean_df[col].astype(float).round(4).astype(str)
        std_rounded = std_df[col].astype(float).round(4).astype(str)
        combined_df[col] = mean_rounded + " ± " + std_rounded
        new_columns[col] = f'{col}\n(Mean ± SD)'
    combined_df.rename(columns=new_columns, inplace=True)
    return combined_df


##############################################
############ Decision Tree ###################
##############################################

# 10 fold stratified cross validation performed with additionally Hyperparameter tuning
# fold is used to keep track of the iterations
def dec_tree(X, y, df_name, clf_name, x_gen, y_gen, gen_df):
    parameters = dict(criterion=['gini', 'entropy'], max_depth=[1, 2, 3, 4, 5, 6], max_features=['sqrt'])
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold = 1
    print(clf_name)

    # Identify numerical columns
    numerical_columns = ['failures']

    # Metrics to store the results
    metric_train = pd.DataFrame(columns=['F1', 'Recall', 'Precision', 'ROC AUC'])
    metric_test = pd.DataFrame(columns=['F1', 'Recall', 'Precision', 'ROC AUC'])
    metric_gen = pd.DataFrame(columns=['F1', 'Recall', 'Precision', 'ROC AUC'])
    feature_importances = np.zeros(X.shape[1])

    # For ROC Curve
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train_index, test_index in cv.split(X, y):
        print('Working on fold ', fold)

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Create a copy of the data to avoid inplace operations
        X_train_sc, X_test_sc = X_train.copy(), X_test.copy()

        # Standardize the data (only numerical one)
        scaler = StandardScaler()
        X_train_sc[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
        X_test_sc[numerical_columns] = scaler.transform(X_test[numerical_columns])

        # Train the model with HP tuning
        cv_in = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        dec_tree = tree.DecisionTreeClassifier(random_state=42)
        clf_GS = GridSearchCV(dec_tree, parameters, cv=cv_in)
        clf_GS.fit(X_train_sc, y_train)
        print('Best Criterion:', clf_GS.best_estimator_.get_params()['criterion'])
        print('Best max_depth:', clf_GS.best_estimator_.get_params()['max_depth'])

        # Calculate metrics via the function eval_performance
        precision_tr, recall_tr, f1_tr, roc_auc_tr, fp_rates_tr, tp_rates_tr = eval_Performance(y_train, X_train_sc,
                                                                                                clf_GS)  # clf_GS takes simply the last or the last model?
        precision_ts, recall_ts, f1_ts, roc_auc_ts, fp_rates_ts, tp_rates_ts = eval_Performance(y_test, X_test_sc,
                                                                                                clf_GS)
        precision_gen, recall_gen, f1_gen, roc_auc_gen, fp_rates_gen, tp_rates_gen = eval_Performance(y_gen, x_gen,
                                                                                                      clf_GS)

        # store the metric in the df previously creted dataframe
        metric_train.loc[fold - 1, 'Precision'] = precision_tr
        metric_train.loc[fold - 1, 'Recall'] = recall_tr
        metric_train.loc[fold - 1, 'F1'] = f1_tr
        metric_train.loc[fold - 1, 'ROC AUC'] = roc_auc_tr

        metric_test.loc[fold - 1, 'Precision'] = precision_ts
        metric_test.loc[fold - 1, 'Recall'] = recall_ts
        metric_test.loc[fold - 1, 'F1'] = f1_ts
        metric_test.loc[fold - 1, 'ROC AUC'] = roc_auc_ts

        # generalization step
        metric_gen.loc[fold - 1, 'Precision'] = precision_gen
        metric_gen.loc[fold - 1, 'Recall'] = recall_gen
        metric_gen.loc[fold - 1, 'F1'] = f1_gen
        metric_gen.loc[fold - 1, 'ROC AUC'] = roc_auc_gen

        # Aggregate feature importances
        feature_importances += clf_GS.best_estimator_.feature_importances_

        # Collect ROC curve data for the fold
        fp_rates, tp_rates, _ = roc_curve(y_test, clf_GS.predict_proba(X_test_sc)[:, 1])
        interp_tpr = np.interp(mean_fpr, fp_rates, tp_rates)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc_score(y_test, clf_GS.predict_proba(X_test_sc)[:, 1]))

        fold += 1

    # Calculate the mean and standard deviation for training, test and foreign (gen) sets
    mean_metrics_train = metric_train.mean().to_frame().T
    std_metrics_train = metric_train.std().to_frame().T
    mean_metrics_test = metric_test.mean().to_frame().T
    std_metrics_test = metric_test.std().to_frame().T
    mean_metrics_gen = metric_gen.mean().to_frame().T
    std_metrics_gen = metric_gen.std().to_frame().T

    # Combine mean and std for train, test, and gen sets
    combined_metrics_train = combine_mean_std(mean_metrics_train, std_metrics_train)
    combined_metrics_test = combine_mean_std(mean_metrics_test, std_metrics_test)
    combined_metrics_gen = combine_mean_std(mean_metrics_gen, std_metrics_gen)

    # Calculate average feature importances
    feature_importances /= fold - 1

    # Create a DataFrame for feature importances
    feature_importances_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importances_df['Feature'], feature_importances_df['Importance'])
    plt.title(f'Feature Importances for {clf_name} - {df_name} dataset')
    plt.xlabel('Feature')
    plt.ylabel('Relative Importance [%]')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'../output/FI_{clf_name}_{df_name}.png')
    plt.show()

    # Plot the tree of the last fold --> for exploratory purposes
    feature = list(X.columns)
    class_n = list(X.index.astype(str))
    plt.figure(figsize=(50, 50))
    tree.plot_tree(clf_GS.best_estimator_,
                   feature_names=feature,
                   class_names=class_n,
                   filled=True)
    plt.title(f'Tree of the last fold-{clf_name}-{df_name}', fontsize=60)
    plt.tight_layout()
    # Save the figure directly into your working directory
    plt.savefig(f'../output/last_fold_{clf_name}_{df_name}.png')
    plt.show()

    # Plot the mean ROC curve --> for exploratory purposes
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.figure(figsize=(6, 6))
    plt.plot(mean_fpr, mean_tpr, color="b", label=f"Mean ROC {clf_name}\n(AUC = {mean_auc:.2f} ± {std_auc:.2f})", lw=2,
             alpha=0.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2, label=r"$\pm$ 1 SD")

    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Classifier', lw=2)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Mean ROC curve with variability\n('{df_name} dataset')")
    plt.legend(loc="lower right")
    plt.show()

    # Rename the index column to "Fold"
    combined_metrics_train.index = [f'{clf_name} {df_name} (train)']
    combined_metrics_test.index = [f'{clf_name} {df_name} (test)']
    combined_metrics_gen.index = [f'{clf_name} {gen_df} (test)']
    # magari aggiungere il clf usato!
    return combined_metrics_train, combined_metrics_test, combined_metrics_gen


X_math_FS = X_maths[chosen_features_maths]
X_portu_FS = X_portu[chosen_features_portu]

dc_math_train, dc_math_test, dc_gen_portu = dec_tree(X_math_FS, y_maths, 'Math', 'Decision Tree',
                                                     X_portu[chosen_features_maths], y_portu, 'Portu in Math')
dc_portu_train, dc_portu_test, dc_gen_math = dec_tree(X_portu_FS, y_portu, 'Portuguese', 'Decision Tree',
                                                      X_maths[chosen_features_portu], y_maths, 'Math in Portu')

# merge all the metrics togheter in a single df
metrics_math = pd.concat([dc_math_train, dc_math_test, dc_gen_portu], axis=0)
print(metrics_math)

metrics_portu = pd.concat([dc_portu_train, dc_portu_test, dc_gen_math], axis=0)
print(metrics_portu)


##############################################
############ Random Forest ###################
##############################################

# Manuel Random Forest, 10 fold stratified cross validation performed for Hyperparameter tuning
# fold is used to keep track of the iterations
def Random_forest(X, y, df_name, clf_name, X_gen, y_gen, gen_name):
    parameters = dict(criterion=['gini', 'entropy'], max_depth=[1, 2, 3, 4, 5, 6], max_features=['sqrt'])
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold = 1
    print(clf_name)

    # Numerical columns to scaled
    num_columns = ['failures']

    # Metrics to store the results
    metrics_train = pd.DataFrame(columns=['F1', 'Recall', 'Precision', 'ROC AUC'])
    metrics_test = pd.DataFrame(columns=['F1', 'Recall', 'Precision', 'ROC AUC'])
    metrics_gen = pd.DataFrame(columns=['F1', 'Recall', 'Precision', 'ROC AUC'])
    feature_importance = np.zeros(X.shape[1])

    # For ROC Curve
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train_index, test_index in cv.split(X, y):
        print('Current fold ', fold)

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Create a copy of the data to avoid inplace operations
        X_train_sc, X_test_sc = X_train.copy(), X_test.copy()

        # Standardize the data (only numerical one)
        scaler = StandardScaler()
        X_train_sc[num_columns] = scaler.fit_transform(X_train[num_columns])
        X_test_sc[num_columns] = scaler.transform(X_test[num_columns])

        # Train the model with HP tuning
        cv_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        rf = RandomForestClassifier(random_state=42)
        clf_GS = GridSearchCV(rf, parameters, cv=cv_inner)
        clf_GS.fit(X_train_sc, y_train)
        print('Best Criterion:', clf_GS.best_estimator_.get_params()['criterion'])
        print('Best max_depth:', clf_GS.best_estimator_.get_params()['max_depth'])

        # Performance metrics calculation with eval_performance function
        precision_train, recall_train, f1_train, roc_auc_train, fp_rates_train, tp_rates_train = eval_Performance(
            y_train, X_train_sc, clf_GS)
        precision_test, recall_test, f1_test, roc_auc_test, fp_rates_test, tp_rates_test = eval_Performance(y_test,
                                                                                                            X_test_sc,
                                                                                                            clf_GS)
        precision_gen, recall_gen, f1_gen, roc_auc_gen, fp_rates_gen, tp_rates_gen = eval_Performance(y_gen, X_gen,
                                                                                                      clf_GS)

        # store the metric in the df previously creted dataframe
        metrics_train.loc[fold - 1, 'Precision'] = precision_train
        metrics_train.loc[fold - 1, 'Recall'] = recall_train
        metrics_train.loc[fold - 1, 'F1'] = f1_train
        metrics_train.loc[fold - 1, 'ROC AUC'] = roc_auc_train

        metrics_test.loc[fold - 1, 'Precision'] = precision_test
        metrics_test.loc[fold - 1, 'Recall'] = recall_test
        metrics_test.loc[fold - 1, 'F1'] = f1_test
        metrics_test.loc[fold - 1, 'ROC AUC'] = roc_auc_test

        # generalization matrix
        metrics_gen.loc[fold - 1, 'Precision'] = precision_gen
        metrics_gen.loc[fold - 1, 'Recall'] = recall_gen
        metrics_gen.loc[fold - 1, 'F1'] = f1_gen
        metrics_gen.loc[fold - 1, 'ROC AUC'] = roc_auc_gen

        # Aggregate feature importances
        feature_importance += clf_GS.best_estimator_.feature_importances_

        # Collect ROC curve data for the fold
        fp_rates, tp_rates, _ = roc_curve(y_test, clf_GS.predict_proba(X_test_sc)[:, 1])
        interp_tpr = np.interp(mean_fpr, fp_rates, tp_rates)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc_score(y_test, clf_GS.predict_proba(X_test_sc)[:, 1]))

        fold += 1

    # Calculate the mean and standard deviation for training, test and foreign (gen) sets
    mean_metrics_train = metrics_train.mean().to_frame().T
    std_metrics_train = metrics_train.std().to_frame().T
    mean_metrics_test = metrics_test.mean().to_frame().T
    std_metrics_test = metrics_test.std().to_frame().T
    mean_metrics_gen = metrics_gen.mean().to_frame().T
    std_metrics_gen = metrics_gen.std().to_frame().T

    # Combine mean and std for train, test, and gen sets
    combined_metrics_train = combine_mean_std(mean_metrics_train, std_metrics_train)
    combined_metrics_test = combine_mean_std(mean_metrics_test, std_metrics_test)
    combined_metrics_gen = combine_mean_std(mean_metrics_gen, std_metrics_gen)

    # Averaging feature importance
    feature_importance /= fold - 1

    # Store feature importances in a DataFrame
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Feature importance plot
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.title(f'Feature Importances {clf_name}- {df_name}')
    plt.xlabel('Feature')
    plt.ylabel('Relative Importance [%]')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'../output/FI_{clf_name}_{df_name}.png')
    plt.show()

    # Get the last estimator (tree) from the random forest
    last_estimator = clf_GS.best_estimator_.estimators_[-1]

    # Plot the tree of the last fold --> for exploratory purposes
    features = list(X.columns)
    class_name = list(X.index.astype(str))
    plt.figure(figsize=(50, 50))
    tree.plot_tree(last_estimator,
                   feature_names=features,
                   class_names=class_name,
                   filled=True)
    plt.title(f'Last tree of the CV fold in-{clf_name}-{df_name} dataset', fontsize=60)
    plt.tight_layout()
    # Save the figure directly into your working directory
    plt.savefig(f'../output/last_fold_{clf_name}_{df_name}.png')
    plt.show()

    # Plot the mean ROC curve --> for exploratory purposes
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.figure(figsize=(6, 6))
    plt.plot(mean_fpr, mean_tpr, color="b", label=f"Mean ROC {clf_name}\n(AUC = {mean_auc:.2f} ± {std_auc:.2f})", lw=2,
             alpha=0.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2, label=r"$\pm$ 1 SD")

    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Classifier', lw=2)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Mean ROC curve with variability\n('{df_name} dataset')")
    plt.legend(loc="lower right")
    plt.show()

    # Rename the index column to "Fold"
    combined_metrics_train.index = [f'{clf_name} {df_name} (train)']
    combined_metrics_test.index = [f'{clf_name} {df_name} (test)']
    combined_metrics_gen.index = [f'{clf_name} {gen_name} (test)']
    # magari aggiungere il clf usato!
    return combined_metrics_train, combined_metrics_test, combined_metrics_gen


# aggiungere classifier magari così da salvarlo nell'output finale
rf_math_train, rf_math_test, rf_gen_portu = Random_forest(X_math_FS, y_maths, 'Math', 'Random Forest',
                                                          X_portu[chosen_features_maths], y_portu, 'Portu in Math')
rf_portu_train, rf_portu_test, rf_gen_math = Random_forest(X_portu_FS, y_portu, 'Portuguese', 'Random Forest',
                                                           X_maths[chosen_features_portu], y_maths, 'Math in Portu')

# merge all the metrics togheter in a single df__modified respect to samu
metrics_math = pd.concat([metrics_math, rf_math_train, rf_math_test, rf_gen_portu], axis=0)
print(metrics_math)

metrics_portu = pd.concat([metrics_portu, rf_portu_train, rf_portu_test, rf_gen_math], axis=0)
print(metrics_portu)

# Save the df as excel file in table format
metrics_math.to_excel('../output/math_metric_DTandRF.xlsx', index_label='Models')
metrics_portu.to_excel('../output/portu_metric_DTandRF.xlsx', index_label='Models')



######################################
######## KNN with CV for MATH ########
######################################

# Defining range for CV
k_range = range(1, 100+int((df_maths.shape[0])**(1/2)))

# List to store the average cross-validated accuracy for each k
cv_scores = []

# Perform 10-fold cross-validation for each value of k
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    kf = KFold(n_splits = 10, shuffle = True, random_state = 42)
    scores = cross_val_score(knn, pd.DataFrame(X_maths_train_scaled[['failures', 'sex_M',
                                                                     'Mjob_health', 'Mjob_services',
                                                                     'Mjob_teacher', 'schoolsup_yes',
                                                                     'higher_yes']]), y_maths_train, cv=kf, scoring = 'f1')
    cv_scores.append(scores.mean())

# Determine the optimal k (the one with the highest cross-validated accuracy)
optimal_k = k_range[np.argmax(cv_scores)]
print(f"The optimal number of neighbors is {optimal_k}")

# Plot the cross-validated accuracy as a function of k
plt.plot(k_range, cv_scores)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Cross-Validated F1 score')
plt.title('KNN Varying number of neighbors\nin the maths dataset')
plt.savefig('../output/CV_maths_KNN.png', dpi = 500)
plt.show()

# Initialize and train the K-NN classifier
knn = KNeighborsClassifier(n_neighbors = optimal_k)
knn.fit(pd.DataFrame(X_maths_train_scaled[['failures', 'sex_M',
                                           'Mjob_health', 'Mjob_services',
                                           'Mjob_teacher', 'schoolsup_yes',
                                           'higher_yes']]), y_maths_train)

# Make train predictions:
y_pred_train = knn.predict(pd.DataFrame(X_maths_train_scaled[['failures', 'sex_M',
                                           'Mjob_health', 'Mjob_services',
                                           'Mjob_teacher', 'schoolsup_yes',
                                           'higher_yes']]))
roc_train = roc_auc_score(y_maths_train, y_pred_train)
print('Math train ROC:', roc_train)
print('Classification train')
print(classification_report(y_maths_train, y_pred_train))

# Make predictions
y_pred = knn.predict(pd.DataFrame(X_maths_test_scaled[['failures', 'sex_M',
                                                       'Mjob_health', 'Mjob_services',
                                                       'Mjob_teacher', 'schoolsup_yes',
                                                       'higher_yes']]))

# Evaluate the model
rocauc_score = roc_auc_score(y_maths_test, y_pred)
print(f'ROC_AUC score: {rocauc_score:.2f}')
print('Classification Report for Maths Dataset:')
print(classification_report(y_maths_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_maths_test, y_pred))

######################################
######## KNN with CV for Port ########
######################################

# Defining range for CV
k_range = range(1, 100+int((df_portu.shape[0])**(1/2)))

# List to store the average cross-validated accuracy for each k
cv_scores = []

# Perform 10-fold cross-validation for each value of k
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    kf = KFold(n_splits = 10, shuffle = True, random_state = 42)
    scores = cross_val_score(knn, pd.DataFrame(X_portu_train_scaled[['failures', 'school_MS', 'sex_M',
                                                                     'famsize_LE3', 'Mjob_health', 'Mjob_teacher',
                                                                     'Fjob_other', 'Fjob_services', 'Fjob_teacher',
                                                                     'reason_reputation', 'guardian_mother', 'guardian_other',
                                                                     'higher_yes']]), y_portu_train, cv = kf, scoring = 'f1')
    cv_scores.append(scores.mean())

# Determine the optimal k (the one with the highest cross-validated accuracy)
optimal_k = k_range[np.argmax(cv_scores)]
print(f"The optimal number of neighbors is {optimal_k}")

# Plot the cross-validated accuracy as a function of k
plt.plot(k_range, cv_scores)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Cross-Validated F1 score')
plt.title('KNN Varying number of neighbors\nin the Portuguese dataset')
plt.savefig('../output/CV_portu_KNN.png', dpi = 500)
plt.show()

# Initialize and train the K-NN classifier
knn = KNeighborsClassifier(n_neighbors = optimal_k)
knn.fit(pd.DataFrame(X_portu_train_scaled[['failures', 'school_MS', 'sex_M',
                                            'famsize_LE3', 'Mjob_health', 'Mjob_teacher',
                                            'Fjob_other', 'Fjob_services', 'Fjob_teacher',
                                            'reason_reputation', 'guardian_mother', 'guardian_other',
                                            'higher_yes']]), y_portu_train)

# Make train predictions:
y_pred_train = knn.predict(pd.DataFrame(X_portu_train_scaled[['failures', 'school_MS', 'sex_M',
                                            'famsize_LE3', 'Mjob_health', 'Mjob_teacher',
                                            'Fjob_other', 'Fjob_services', 'Fjob_teacher',
                                            'reason_reputation', 'guardian_mother', 'guardian_other',
                                            'higher_yes']]))
roc_train = roc_auc_score(y_portu_train, y_pred_train)
print('Portu train ROC:', roc_train)
print('Classification train')
print(classification_report(y_portu_train, y_pred_train))


# Make test predictions
y_pred = knn.predict(pd.DataFrame(X_portu_test_scaled[['failures', 'school_MS', 'sex_M',
                                                         'famsize_LE3', 'Mjob_health', 'Mjob_teacher',
                                                         'Fjob_other', 'Fjob_services', 'Fjob_teacher',
                                                         'reason_reputation', 'guardian_mother', 'guardian_other',
                                                         'higher_yes']]))

# Evaluate the model
rocauc_score = roc_auc_score(y_portu_test, y_pred)
print(f'ROC_AUC score: {rocauc_score:.2f}')
print('Classification Report for Portuguese dataset:')
print(classification_report(y_portu_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_portu_test, y_pred))