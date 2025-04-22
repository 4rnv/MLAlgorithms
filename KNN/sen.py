import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score

def most_common(labels):
    most_common_label = ""
    max_count = 0
    count = {}
    for i in labels:
        if i not in count:
            count[i] = 0
        count[i] += 1
        if count[i]>max_count:
            max_count = count[i]
            most_common_label = i
    return most_common_label

class KNNClassifier:
    def __init__(self):
        self.data_points = []
        self.labels = []

    def load(self,data, labels):
        # for datapoint in data:
        self.data_points = data.to_numpy()
        self.labels = labels.tolist() 

    def classify(self, input, k):
        distances = []
        input = np.array(input)
        for i, data_point in enumerate(self.data_points):
            distance = np.linalg.norm(np.subtract(input, data_point))
            distances.append((distance, self.labels[i]))
        nearest_neighbors = sorted(distances, key=lambda x: x[0])[:k]
        neighbour_classifications = [neighbor[1] for neighbor in nearest_neighbors]
        return most_common(neighbour_classifications)

    def predict(self, test_data, test_labels, k_s):
        results = {}
        for k in k_s:
            predictions = []
            for index, row in test_data.iterrows():
                input_color = row[['R', 'G', 'B']].values
                predicted_label = self.classify(input_color, k)
                predictions.append(predicted_label)
            accuracy = accuracy_score(test_labels, predictions)
            precision = precision_score(test_labels, predictions, average='weighted', zero_division=0)
            f1 = f1_score(test_labels, predictions, average='weighted', zero_division=0)
            results[k] = {
                'accuracy': accuracy,
                'precision': precision,
                'f1_score': f1
            }
        
        return results

classifier = KNNClassifier()
data = pd.read_csv('train.csv')
classifier.load(data.iloc[:,:-1],data.iloc[:,-1])

test_data = pd.read_csv('test.csv')
test_labels = test_data['Label']
k_values = [1, 5, 10, 20, 50, 100]
results = classifier.predict(test_data, test_labels, k_values)
for k, metrics in results.items():
    print(f"k={k}: Accuracy={metrics['accuracy']:.2f}, Precision={metrics['precision']:.2f}, F1 Score={metrics['f1_score']:.2f}")