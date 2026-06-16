from sklearn.linear_model import LinearRegression

# Sample training data
X = [[2], [4], [6], [8]]
y = [65, 75, 85, 95]

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Make prediction
prediction = model.predict([[10]])

print("Predicted Score:", prediction[0])
