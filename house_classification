import os
import requests
import sys
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import TargetEncoder
from sklearn.metrics import classification_report

if __name__ == '__main__':
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if 'house_class.csv' not in os.listdir('../Data'):
        sys.stderr.write("[INFO] Dataset is loading.\n")
        url = "https://www.dropbox.com/s/7vjkrlggmvr5bc1/house_class.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/house_class.csv', 'wb').write(r.content)
        sys.stderr.write("[INFO] Loaded.\n")

    # write your code here

# load data frame
df = pd.read_csv('../Data/house_class.csv')
#'/Users/jb/PycharmProjects/House Classification/House Classification/Data/house_class.csv')

#print(df)

X, y = df.iloc[:, 1:], df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=X['Zip_loc'].values,test_size=0.30, random_state=1)

#print(dict(X_train['Zip_loc'].value_counts().items()))
x_cat = X_train[['Zip_area', 'Zip_loc','Room']]
x_cat_test = X_test[['Room','Zip_area', 'Zip_loc']]

'''OneHotEncoder Section - ohe'''
# Transforming with OneHotEncoder Room, Zip Area and Zip Loc using Column Transformer
ct_ohe = ColumnTransformer(transformers = [('Zip_area',OneHotEncoder(drop='first'),['Zip_area']),
                                        ('Zip_loc',OneHotEncoder(drop='first'),['Zip_loc']),
                                     ('Room',OneHotEncoder(drop='first'),['Room'])])

# fit encoder to train data
encoder_ohe = ct_ohe.fit(x_cat)


# transform train and test data
X_train_transformed_ohe = pd.DataFrame(encoder_ohe.transform(X_train[['Zip_area', 'Zip_loc', 'Room']]).toarray(), index=X_train.index)

X_test_transformed_ohe = pd.DataFrame(encoder_ohe.transform(X_test[['Zip_area', 'Zip_loc', 'Room']]).toarray(), index=X_test.index)

# Join Data to Area, Lon and Lat for train and test data
X_train_final_ohe = X_train[['Area', 'Lon', 'Lat']].join(X_train_transformed_ohe).add_prefix('enc')

X_test_final_ohe = X_test[['Area', 'Lon', 'Lat']].join(X_test_transformed_ohe).add_prefix('enc')

#Create Decision Tree Classifier
clf_ohe = DecisionTreeClassifier(criterion='entropy',max_features=3,splitter='best',max_depth=6,min_samples_split=4,random_state=3)

#Fit the model to train data and predict with test data
clf_train = clf_ohe.fit(X_train_final_ohe, y_train)

y_hat_ohe = clf_ohe.predict(X_test_final_ohe)

'''Ordinal Encoder Section oe'''
# Transforming with Ordinalencouder Room, Zip Area and Zip Loc using Column Transformer
ct_oe = ColumnTransformer(transformers = [('Zip_area',OrdinalEncoder(),['Zip_area']),
                                        ('Zip_loc',OrdinalEncoder(),['Zip_loc']),
                                     ('Room',OrdinalEncoder(),['Room'])])

# fit encoder to train data
encoder_oe = ct_oe.fit(x_cat)


# transform train and test data
X_train_transformed_oe = pd.DataFrame(encoder_oe.transform(X_train[['Zip_area', 'Zip_loc', 'Room']]), index=X_train.index)

X_test_transformed_oe = pd.DataFrame(encoder_oe.transform(X_test[['Zip_area', 'Zip_loc', 'Room']]), index=X_test.index)

# Join Data to Area, Lon and Lat for train and test data
X_train_final_oe = X_train[['Area', 'Lon', 'Lat']].join(X_train_transformed_oe).add_prefix('enc')

X_test_final_oe = X_test[['Area', 'Lon', 'Lat']].join(X_test_transformed_oe).add_prefix('enc')

#Create Decision Tree Classifier
clf_oe = DecisionTreeClassifier(criterion='entropy',max_features=3,splitter='best',max_depth=6,min_samples_split=4,random_state=3)

#Fit the model to train data and predict with test data
clf_train_oe = clf_oe.fit(X_train_final_oe, y_train)

y_hat_oe = clf_oe.predict(X_test_final_oe)

'''Target Encoder Section te '''

encoder_te = TargetEncoder(cols=['Zip_area','Zip_loc','Room'])

encoder_te.fit(X_train, y_train)

X_train_encoded_te = encoder_te.transform(X_train)
X_test_encoded_te = encoder_te.transform(X_test)

clf_te = DecisionTreeClassifier(criterion='entropy',max_features=3,splitter='best',max_depth=6,min_samples_split=4,random_state=3)

clf_train_te = clf_te.fit(X_train_encoded_te, y_train)

y_hat_te = clf_te.predict(X_test_encoded_te)

#classification report
y_review_ohe=classification_report(y_test,y_hat_ohe,output_dict=True)
y_review_oe=classification_report(y_test,y_hat_oe,output_dict=True)
y_review_te=classification_report(y_test,y_hat_te,output_dict=True)

print(f"OneHotEncoder:{round(y_review_ohe['macro avg']['f1-score'],2)}")
print(f"OrdinalEncoder:{round(y_review_oe['macro avg']['f1-score'],2)}")
print(f"TargetEncoder:{round(y_review_te['macro avg']['f1-score'],2)}")
