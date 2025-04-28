import pandas as pd

df = pd.read_csv('naivebayes.csv')
df.fillna(method='ffill', inplace=True)
df.head()

class_counts = df['class'].value_counts()
class_dict = class_counts.to_dict()

feature_dicts = {}
for column in df.columns[:-1]:
    feature_dicts[column] = df[column].value_counts().to_dict()

for feature, counts in feature_dicts.items():
    print(f"{feature} counts: {counts}")

def prediction(features : dict):
    prob_yes = class_dict['yes']
    prob_no = class_dict['no']
    print(feature_dicts)
    for feature, value in features.items():
        print(value)
        if value in feature_dicts[feature]:
            prob_yes *= (df[(df[feature] == value) & (df['class'] == 'yes')].shape[0] + 1) / (class_dict['yes'] + len(feature_dicts[feature]))
            prob_no *= (df[(df[feature] == value) & (df['class'] == 'no')].shape[0] + 1) / (class_dict['no'] + len(feature_dicts[feature]))

    return prob_yes, prob_no

features = {}
for column in df.columns[:-1]:
    features[column] = input(f"Enter {column}: ").lower()

probability_yes, probability_no = prediction(features)
print("\nProbability of yes:", probability_yes, "\nProbability of no:", probability_no)

if probability_yes > probability_no:
    print("Probability of Yes is higher")
else:
    print("Probability of No is higher")