import csv
import math

data = []
with open('sample_logistic_data.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        data.append(row)

features = []
labels = []

for row in data:
    budget_variance = float(row['budget_variance'])
    timeline_delay_days = float(row['timeline_delay_days'])
    team_size = float(row['team_size'])
    complexity_score = float(row['complexity_score'])
    risk_flag = float(row['risk_flag'])
    
    features.append([budget_variance, timeline_delay_days, team_size, complexity_score])
    labels.append(risk_flag)

num_samples = len(features)
num_features = len(features[0])

weights = [0.0] * num_features
bias = 0.0
learning_rate = 0.01
iterations = 1000

for iteration in range(iterations):
    total_loss = 0
    
    for i in range(num_samples):
        z = bias
        for j in range(num_features):
            z = z + weights[j] * features[i][j]
        
        prediction = 1 / (1 + math.exp(-z))
        
        error = prediction - labels[i]
        total_loss = total_loss + (labels[i] * math.log(prediction + 1e-10) + (1 - labels[i]) * math.log(1 - prediction + 1e-10))
        
        bias = bias - learning_rate * error
        
        for j in range(num_features):
            weights[j] = weights[j] - learning_rate * error * features[i][j]
    
    total_loss = -total_loss / num_samples
    
    if (iteration + 1) % 100 == 0:
        print(f"Iteration {iteration + 1}, Loss: {total_loss:.4f}")

print("\nTrained Weights:")
print(f"Budget Variance: {weights[0]:.4f}")
print(f"Timeline Delay Days: {weights[1]:.4f}")
print(f"Team Size: {weights[2]:.4f}")
print(f"Complexity Score: {weights[3]:.4f}")
print(f"Bias: {bias:.4f}")

print("\nPredictions on Training Data:")
for i in range(num_samples):
    z = bias
    for j in range(num_features):
        z = z + weights[j] * features[i][j]
    
    prediction = 1 / (1 + math.exp(-z))
    predicted_class = 1 if prediction >= 0.5 else 0
    
    print(f"Sample {i + 1}: Probability={prediction:.4f}, Predicted Class={predicted_class}, Actual Class={int(labels[i])}")
