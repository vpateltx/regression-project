import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("youtube recommendation dataset.csv")

# Features
X = data[["video_duration"]]

# Target
y = data["watch_time"]

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict watch time for a 1000-second video
prediction = model.predict(pd.DataFrame([[1000]], columns=["video_duration"]))

print("Predicted Watch Time:")
print(prediction[0])

