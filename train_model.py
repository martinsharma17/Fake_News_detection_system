import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from scipy.sparse import csr_matrix
import joblib

class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, tolerance=1e-4):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        """Train the logistic regression model"""
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # Gradient descent
        for iteration in range(self.max_iter):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = np.mean(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check for convergence
            if iteration > 0 and np.mean(np.abs(dw)) < self.tolerance:
                print(f"Converged at iteration {iteration}")
                break
                
        print(f"Training completed in {iteration + 1} iterations")
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if self.weights is None or self.bias is None:
            raise ValueError("Model must be fitted before making predictions")
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)
    
    def predict(self, X, threshold=0.5):
        """Predict binary classes"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

# Load datasets
fake = pd.read_csv('fake.csv')
true = pd.read_csv('true.csv')

# Add labels: 0 for fake, 1 for true
fake['label'] = 0
true['label'] = 1

# Combine datasets
all_news = pd.concat([fake, true], ignore_index=True)

# Combine title and text
all_news['content'] = all_news['title'].fillna('') + ' ' + all_news['text'].fillna('')

# Features and labels
X = all_news['content']
y = all_news['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Convert sparse matrices to dense arrays for our custom implementation
# Convert sparse matrices to dense arrays
X_train_dense = np.asarray(X_train_vec.todense())  # type: ignore
X_test_dense = np.asarray(X_test_vec.todense())  # type: ignore

# Train custom model
print("Training custom logistic regression model...")
model = CustomLogisticRegression(learning_rate=0.01, max_iter=1000)
model.fit(X_train_dense, y_train.values)  # type: ignore

# Evaluate
y_pred = model.predict(X_test_dense)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
print('Model and vectorizer saved!') 