#dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load dataset
def load_dataset():
    data = {
        'Pclass': [1, 1, 1, 1, 1],
        'Sex': ['male', 'female', 'female', 'male', 'male'],
        'Age': [22, 38, 26, 35, 35],
        'SibSp': [1, 1, 0, 0, 0],
        'Parch': [0, 0, 0, 0, 0],
        'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05],
        'Embarked': ['S', 'C', 'S', 'S', 'S'],
        'Survived': [0, 1, 1, 0, 0]
    }
    df = pd.DataFrame(data)

    # Convert categorical variables to numerical values
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    return df

# Train model
def train_model(df):
    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test

# Evaluate model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# Main function
def main():
    df = load_dataset()
    model, X_test, y_test = train_model(df)
    accuracy = evaluate_model(model, X_test, y_test)
    print("Accuracy:", accuracy)
if __name__ == "__main__":
    main()
