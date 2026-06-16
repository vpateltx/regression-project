import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

data = pd.read_csv("youtube recommendation dataset.csv")

X = data[["video_duration"]]
y = data["watch_time"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

new_video = pd.DataFrame([[1000]], columns=["video_duration"])
predicted_watch_time = model.predict(new_video)

print("YouTube Watch Time Regression Model")
print("----------------------------------")
print("R² Score:", r2)
print("Mean Absolute Error:", mae)
print("Predicted watch time for a 1000-second video:", predicted_watch_time[0])
