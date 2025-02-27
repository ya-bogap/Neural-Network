Immunotherapy Treatment Prediction Using Neural Networks
# Overview
This project uses deep learning (Neural Networks) to predict the success of immunotherapy treatment for warts based on patient information. We use Keras (TensorFlow) to build and train a neural network with two hidden layers using the tanh activation function.

The model is trained on 80 patients and tested on 10 new patients to evaluate its predictive power.

# Objective
The goal of this project is to:

Train a neural network to predict whether a patient’s immunotherapy treatment will be successful.
Use patient features (such as age, sex, number of warts, etc.) to determine the likelihood of a successful treatment.
Evaluate the model's accuracy and analyze predictions on unseen test data.
# Dataset Description
The dataset is stored in Immunotherapy.xlsx, containing two sheets:

Training data set → 80 patients (used for training the model)
Test data set → 10 patients (used to check the model’s accuracy)

# Features (Input Variables)
These are the factors used to predict treatment success:

Sex (1 = Male, 0 = Female)
Age (Years)
Time (Duration of treatment)
Number_of_Warts (Count of warts)
Type (Wart type classification)
Area (Size of affected area)
Induration_diameter (Diameter of skin hardening)
# Target Variable (Output)
Result_of_Treatment (1 = Successful, 0 = Unsuccessful)
# Steps and Explanations
# Step 1: Load the Data
We load the dataset from Excel and split it into training and testing sets.

train_data = pd.read_excel("Immunotherapy.xlsx", sheet_name="Training data set")
test_data = pd.read_excel("Immunotherapy.xlsx", sheet_name="Test data set")
# Step 2: Prepare Features and Target Variable
We extract the features (X) and target variable (y):

X_train = train_data.drop(columns=["Result_of_Treatment"]).values
y_train = train_data["Result_of_Treatment"].values

X_test = test_data.drop(columns=["Result_of_Treatment"]).values
y_test = test_data["Result_of_Treatment"].values

# Step 3: Normalize the Data
We scale the input features using StandardScaler():

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Build the Neural Network
We create a Sequential model with:

Two hidden layers (16 and 8 neurons, tanh activation)
One output layer (1 neuron, sigmoid activation for classification)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(16, activation='tanh', input_shape=(X_train.shape[1],)),  # First hidden layer
    Dense(8, activation='tanh'),  # Second hidden layer
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Step 5: Compile the Model
We use the Adam optimizer and Binary Crossentropy loss:

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

Adam optimizer speeds up training with adaptive learning rates.
Binary Crossentropy measures error for classification problems (0 or 1).
# Step 6: Train the Model
We train the model for 100 epochs with a batch size of 8.

history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=1)

Epochs = 100 → The model sees the data 100 times to learn patterns.
Batch size = 8 → Trains the model in small groups to improve efficiency.
# Step 7: Evaluate the Model
We check the accuracy on test data:

test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Step 8: Make Predictions
We generate predictions for the 10 test patients:

y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary values
print("Predicted Treatment Results:", y_pred.squeeze())


# Results
Test Accuracy:

Test Accuracy: 0.4000
The model correctly predicts 40% of test cases.
It can be improved with better feature selection, tuning, or more data.
Predicted Treatment Results (Test Set):

[1 1 0 1 0 0 0 0 0 1]
1 = Treatment successful
0 = Treatment unsuccessful
