import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
#import matplotlib.pyplot as plt

file_path = 'StudentsPerformance.csv'
data = pd.read_csv(file_path)

# Direct mapping of boolean features to 0 and 1 for clarity
data['gender'] = data['gender'].map({'male': 1, 'female': 0})
data['lunch'] = data['lunch'].map({'standard': 1, 'free/reduced': 0})
data['test preparation course'] = data['test preparation course'].map({'completed': 1, 'none': 0})


data = pd.get_dummies(data, columns=['race/ethnicity', 'parental level of education'])

feature_columns = data.columns[:-3]

X_target1 = data[feature_columns]
y_target1 = data['math score']

X_target2 = data[feature_columns]
y_target2 = data['reading score']

X_target3 = data[feature_columns]
y_target3 = data['writing score']

X_train, X_test, y_train, y_test = train_test_split(X_target1, y_target1, test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_target2, y_target2, test_size=0.2, random_state=43)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X_target3, y_target3, test_size=0.2, random_state=44)

X_train = X_train.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
X_test = X_test.astype('float32')
X_train2 = X_train.astype('float32')
y_train2 = y_train.astype('float32')
y_test2 = y_test.astype('float32')
X_test2 = X_test.astype('float32')
X_train3 = X_train.astype('float32')
y_train3 = y_train.astype('float32')
y_test3 = y_test.astype('float32')
X_test3 = X_test.astype('float32')

model = Sequential([
    Dense(28, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(14, activation='relu'),
    Dense(1)
])
model2 = Sequential([
    Dense(28, activation='relu', input_shape=(X_train2.shape[1],)),
    Dense(14, activation='relu'),
    Dense(1)
])
model3 = Sequential([
    Dense(28, activation='relu', input_shape=(X_train3.shape[1],)),
    Dense(14, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.1),
              loss='mse',
              metrics=['mae'])

model2.compile(optimizer=Adam(learning_rate=0.1),
              loss='mse',
              metrics=['mae'])

model3.compile(optimizer=Adam(learning_rate=0.1),
              loss='mse',
              metrics=['mae'])

history = model.fit(X_train, y_train, epochs=145, batch_size=32, validation_split=0.2)
history2 = model2.fit(X_train2, y_train2, epochs=145, batch_size=32, validation_split=0.2)
history3 = model3.fit(X_train3, y_train3, epochs=145, batch_size=32, validation_split=0.2)

model.save('model_math.keras')
model2.save('model_reading.keras')
model3.save('model_writing.keras')