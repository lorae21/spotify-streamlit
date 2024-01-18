import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import LabelEncoder


# random seed
seed = 42

df = pd.read_csv("/workspaces/spotify-streamlit/spotify_songs.csv")

df.sample(frac=1, random_state=seed)

# Print column names
print("Column Names:", df.columns)

# Clean up column names
df.columns = df.columns.str.strip()

# Verify column names again
print("Cleaned Column Names:", df.columns)

print(df.dtypes)

numeric_columns = ['danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

for column in numeric_columns:
    print(f"Unique values in {column}: {df[column].unique()}")

# Select features
X = df[['danceability', 'energy', 'key', 'loudness', 'speechiness','acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]

# selecting features and target data
X = df[[ 'danceability', 'energy', 'key', 'loudness', 'speechiness','acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]

# Encode the categorical target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['playlist_genre'])


# Numerical values assigned to playlist genres
playlist_genres_encoded_values = label_encoder.classes_

# Display the numerical values assigned to playlist genres
print("Numerical values assigned to playlist genres:", playlist_genres_encoded_values)

# split data into train and test sets
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed, stratify=y)

# create an instance of the random forest classifier
clf = RandomForestClassifier(n_estimators=100)

# train the classifier on the training data
clf.fit(X_train, y_train)

# predict on the test set
y_pred = clf.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# save the model to disk
joblib.dump(clf, "rf_model.sav")