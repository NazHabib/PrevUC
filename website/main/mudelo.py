import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


file_path = 'StudentsPerformance.csv'
data = pd.read_csv(file_path)

data['gender'] = data['gender'].map({'male': 1, 'female': 0})
data['lunch'] = data['lunch'].map({'standard': 1, 'free/reduced': 0})
data['test preparation course'] = data['test preparation course'].map({'completed': 1, 'none': 0})
data = pd.get_dummies(data, columns=['race/ethnicity', 'parental level of education'])

feature_columns = data.columns.drop(['math score', 'reading score', 'writing score'])
X_target1 = data[feature_columns].astype('int32')
y_target1 = data['math score'].astype('int32')
X_target2 = data[feature_columns].astype('int32')
y_target2 = data['reading score'].astype('int32')
X_target3 = data[feature_columns].astype('int32')
y_target3 = data['writing score'].astype('int32')

X_train, X_test, y_train, y_test = train_test_split(X_target1, y_target1, test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_target2, y_target2, test_size=0.2, random_state=43)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X_target3, y_target3, test_size=0.2, random_state=44)

math_architecture = [6, 61, 32]
reading_architecture = [6, 36, 32]
writing_architecture = [1, 56]

def build_model(architecture, input_shape):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    for units in architecture:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(1))
    return model

model = build_model(math_architecture, X_train.shape[1])
model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=79, batch_size=11, validation_split=0.2, verbose=0)

model2 = build_model(reading_architecture, X_train2.shape[1])
model2.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error', metrics=['mae'])
history2 = model2.fit(X_train2, y_train2, epochs=30, batch_size=41, validation_split=0.2, verbose=0)

model3 = build_model(writing_architecture, X_train3.shape[1])
model3.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error', metrics=['mae'])
history3 = model3.fit(X_train3, y_train3, epochs=59, batch_size=30, validation_split=0.2, verbose=0)

model.save('model_math.keras')
model2.save('model_reading.keras')
model3.save('model_writing.keras')