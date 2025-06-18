# Breast Cancer Detection using KNN and SVM
# Updated for Python 3 and modern scikit-learn
# Designed to work in breast-cancer subfolder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import warnings
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from pandas.plotting import scatter_matrix

# Add parent directory to path to access shared resources
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def get_project_root():
    """Get the project root directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)

def download_data():
    """Download and load the Wisconsin Breast Cancer dataset"""
    project_root = get_project_root()
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    data_file = os.path.join(data_dir, 'breast-cancer-wisconsin.data')
    
    try:
        # Check if file already exists
        if os.path.exists(data_file):
            print("Dataset already exists, loading from file...")
        else:
            print("Downloading breast cancer dataset...")
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
            
            response = requests.get(url)
            response.raise_for_status()
            
            # Save to local file
            with open(data_file, 'w') as f:
                f.write(response.text)
            print("Dataset downloaded successfully!")
        
        # Column names for the dataset
        names = ['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
                'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
                'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
        
        df = pd.read_csv(data_file, names=names)
        print("Dataset loaded successfully!")
        return df
        
    except requests.RequestException as e:
        print(f"Error downloading dataset: {e}")
        print("Please check your internet connection and try again.")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def preprocess_data(df):
    """Preprocess the breast cancer dataset"""
    print("\n=== Data Preprocessing ===")
    print("Running from breast-cancer folder")
    
    # Display basic information
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Handle missing values (marked as '?' in this dataset)
    df.replace('?', np.nan, inplace=True)
    print(f"Rows with '?' values: {df.isnull().sum().sum()}")
    
    # Remove the ID column as it's not useful for prediction
    df.drop(['id'], axis=1, inplace=True)
    
    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing values
    df.dropna(inplace=True)
    print(f"Final dataset shape after cleaning: {df.shape}")
    
    # Display class distribution
    print(f"\nClass distribution:")
    print(f"Benign (2): {(df['class'] == 2).sum()}")
    print(f"Malignant (4): {(df['class'] == 4).sum()}")
    
    return df

def explore_data(df):
    """Perform exploratory data analysis"""
    print("\n=== Exploratory Data Analysis ===")
    
    # Basic statistics
    print("Dataset description:")
    print(df.describe())
    
    try:
        # Correlation matrix
        plt.figure(figsize=(12, 10))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        # Feature distributions
        plt.figure(figsize=(15, 12))
        df.hist(figsize=(15, 12), bins=20)
        plt.suptitle('Feature Distributions')
        plt.tight_layout()
        plt.show()
        
        # Scatter plot matrix (subset of features due to size)
        important_features = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape', 'class']
        if len(df.columns) > 4:
            scatter_matrix(df[important_features], figsize=(12, 12), alpha=0.7)
            plt.suptitle('Scatter Plot Matrix (Key Features)')
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"Note: Visualization error (continuing without plots): {e}")

def prepare_features(df):
    """Prepare features and target variables"""
    print("\n=== Feature Preparation ===")
    
    # Separate features and target
    X = df.drop(['class'], axis=1).values
    y = df['class'].values
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_models(X_train, y_train):
    """Train and compare KNN and SVM models"""
    print("\n=== Model Training and Comparison ===")
    
    # Define models
    models = {
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    # Perform cross-validation
    results = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        results[name] = cv_scores
        print(f"{name} - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return models, results

def optimize_hyperparameters(X_train, y_train):
    """Optimize hyperparameters for both models"""
    print("\n=== Hyperparameter Optimization ===")
    print("This will take a few minutes...")
    
    # KNN hyperparameter tuning
    knn = KNeighborsClassifier()
    knn_params = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    print("Optimizing KNN parameters...")
    knn_grid = GridSearchCV(knn, knn_params, cv=5, scoring='accuracy', n_jobs=-1)
    knn_grid.fit(X_train, y_train)
    
    print(f"Best KNN parameters: {knn_grid.best_params_}")
    print(f"Best KNN CV score: {knn_grid.best_score_:.4f}")
    
    # SVM hyperparameter tuning
    svm = SVC(probability=True, random_state=42)
    svm_params = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.001, 0.01]
    }
    
    print("Optimizing SVM parameters...")
    svm_grid = GridSearchCV(svm, svm_params, cv=5, scoring='accuracy', n_jobs=-1)
    svm_grid.fit(X_train, y_train)
    
    print(f"Best SVM parameters: {svm_grid.best_params_}")
    print(f"Best SVM CV score: {svm_grid.best_score_:.4f}")
    
    return knn_grid.best_estimator_, svm_grid.best_estimator_

def evaluate_models(models, X_test, y_test):
    """Evaluate models on test set"""
    print("\n=== Model Evaluation ===")
    
    results = {}
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'accuracy': accuracy,
            'predictions': y_pred
        }
        
        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        try:
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Benign', 'Malignant'],
                       yticklabels=['Benign', 'Malignant'])
            plt.title(f'{name} - Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.show()
        except Exception as e:
            print(f"Note: Visualization error: {e}")
    
    return results

def make_prediction(model, scaler, example_data=None):
    """Make prediction for a new patient"""
    print("\n=== Making Predictions ===")
    
    if example_data is None:
        # Example patient data: [clump_thickness, uniform_cell_size, uniform_cell_shape,
        # marginal_adhesion, single_epithelial_size, bare_nuclei, bland_chromatin, 
        # normal_nucleoli, mitoses]
        example_data = [4, 2, 1, 1, 1, 2, 3, 2, 1]
    
    # Ensure example_data is the right shape and standardize
    example_array = np.array(example_data).reshape(1, -1)
    example_scaled = scaler.transform(example_array)
    
    # Make prediction
    prediction = model.predict(example_scaled)[0]
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(example_scaled)[0]
        print(f"Prediction probabilities: Benign: {probabilities[0]:.4f}, Malignant: {probabilities[1]:.4f}")
    
    print(f"Example patient data: {example_data}")
    
    if prediction == 2:
        print("Prediction: Benign (Non-cancerous)")
    elif prediction == 4:
        print("Prediction: Malignant (Cancerous)")
    
    return prediction

def save_model(model, scaler, model_name):
    """Save the trained model and scaler"""
    project_root = get_project_root()
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    import pickle
    model_path = os.path.join(models_dir, f'breast_cancer_{model_name.lower()}_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(models_dir, f'breast_cancer_{model_name.lower()}_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")

def main():
    """Main function to run the breast cancer detection pipeline"""
    print("=== Breast Cancer Detection System ===")
    print("Running from breast-cancer folder")
    print("=" * 50)
    
    # Download and load data
    df = download_data()
    if df is None:
        return
    
    # Preprocess data
    df_clean = preprocess_data(df)
    
    # Explore data
    explore_data(df_clean)
    
    # Prepare features
    X_train, X_test, y_train, y_test, scaler = prepare_features(df_clean)
    
    # Train basic models
    models, cv_results = train_models(X_train, y_train)
    
    # Ask user if they want to perform hyperparameter optimization
    print("\nWould you like to perform hyperparameter optimization?")
    print("Note: This will take a few minutes but may improve accuracy.")
    choice = input("Enter 'y' for yes, 'n' for no (default): ").lower().strip()
    
    if choice == 'y':
        # Optimize hyperparameters
        best_knn, best_svm = optimize_hyperparameters(X_train, y_train)
        optimized_models = {'KNN (Optimized)': best_knn, 'SVM (Optimized)': best_svm}
    else:
        # Use default models
        models['KNN'].fit(X_train, y_train)
        models['SVM'].fit(X_train, y_train)
        optimized_models = models
    
    # Evaluate models
    results = evaluate_models(optimized_models, X_test, y_test)
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_model = optimized_models[best_model_name]
    
    print(f"\nBest performing model: {best_model_name}")
    print(f"Best accuracy: {results[best_model_name]['accuracy']:.4f}")
    
    # Make example prediction
    make_prediction(best_model, scaler)
    
    # Save best model
    save_model(best_model, scaler, best_model_name.split()[0])
    
    print("\n=== Analysis Complete ===")
    print("Check the 'models' folder in your main project directory for saved files.")

if __name__ == "__main__":
    main()