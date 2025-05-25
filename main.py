import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import metrics

dataset = pd.read_excel("crp.xlsx")
dataset

dataset.info()

fig, ax = plt.subplots(11, 1, figsize=(10, 50))

for i in range(0, 11):
    ax[i].barh(dataset['City'], dataset[dataset.columns[i+2]], 0.6, color='Salmon')
    ax[i].set_title('City vs ' + dataset.columns[i+2])
plt.show()

# Ensure City column is treated as string
dataset['City'] = dataset['City'].astype(str)

# Create the new dataframe
new_df = pd.DataFrame(columns=['Year', 'City', 'Population (in Lakhs) (2011)+', 'Number Of Cases', 'Type'])

# Loop through the relevant columns
for i in range(3, 13):
    temp = dataset[['Year', 'City', 'Population (in Lakhs) (2011)+']].copy()
    temp['Number Of Cases'] = dataset.iloc[:, i]
    temp['Type'] = dataset.columns[i]

    new_df = pd.concat([new_df, temp], ignore_index=True)

# Optional: reset index and check
new_df.reset_index(drop=True, inplace=True)
print(new_df.head())
print(new_df.dtypes)

new_df

new_df['Crime Rate'] = new_df['Number Of Cases'] / new_df['Population (in Lakhs) (2011)+']
new_df

new_df = new_df.drop(['Number Of Cases'], axis=1)
new_df

# saving the new dataset as an excel file
new_df.to_excel("new_dataset.xlsx", index=False)

new_dataset = pd.read_excel("new_dataset.xlsx")
new_dataset

new_dataset.info()

new_dataset.describe()

new_dataset['City'] = new_dataset['City'].astype(str)
le = LabelEncoder()
new_dataset['City'] = le.fit_transform(new_dataset['City'])
mapping = dict(zip(le.classes_, range(len(le.classes_))))
# Saving the mapping file for further use
file = open('City_Mapping.txt', 'wt')
for key,val in mapping.items():
    print(str(key) + " - " + str(val) + '\n')
    file.write(str(key) + " - " + str(val) + '\n')

    new_dataset['Type'] = le.fit_transform(new_dataset['Type'])
    mapping = dict(zip(le.classes_, range(len(le.classes_))))
# Saving the mapping file for further use
file = open('Type_Mapping.txt', 'wt')
for key,val in mapping.items():
    print(str(key) + " - " + str(val) + '\n')
    file.write(str(key) + " - " + str(val) + '\n')

    new_dataset

    x = new_dataset[new_dataset.columns[0:4]].values
    x
    y = new_dataset['Crime Rate'].values
    y
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=50)
    x_train

    y_train



    model1 = svm.SVR()
    model1.fit(x_train, y_train)
    y_pred = model1.predict(x_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('R2 score:', metrics.r2_score(y_test, y_pred))

    model3 = tree.DecisionTreeRegressor()
    model3.fit(x_train, y_train)
    y_pred = model3.predict(x_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('R2 score:', metrics.r2_score(y_test, y_pred))

    model4 = RandomForestRegressor(random_state=0)
    model4.fit(x_train, y_train)
    y_pred = model4.predict(x_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('R2 score:', metrics.r2_score(y_test, y_pred))



    import pickle

    # saving the model as .pkl file
    pkl_filename = "model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model4, file)

        # checking the saved model accuracy
        with open(pkl_filename, 'rb') as file:
            pickle_model = pickle.load(file)
        score = pickle_model.score(x_test, y_test)
        print(score)


