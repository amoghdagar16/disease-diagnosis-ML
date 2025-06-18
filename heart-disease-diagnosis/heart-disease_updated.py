# Heart Disease Diagnosis using Neural Networks
# Updated for Python 3 and modern TensorFlow/Keras
# Designed to work in heart-disease-diagnosis subfolder

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import warnings
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Add parent directory to path to access shared resources
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def get_project_root():
    """Get the project root directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)

def download_data():
    """Download and load the Cleveland Heart Disease dataset"""
    project_root = get_project_root()
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    data_file = os.path.join(data_dir, 'cleveland-heart-disease.data')
    
    try:
        # Check if file already exists
        if os.path.exists(data_file):
            print("Dataset already exists, loading from file...")
        else:
            print("Downloading heart disease dataset...")
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
            
            response = requests.get(url)
            response.raise_for_status()
            
            # Save to local file
            with open(data_file, 'w') as f:
                f.write(response.text)
            print("Dataset downloaded successfully!")
        
        # Column names for the dataset
        names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'class']
        
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
    """Preprocess the heart disease dataset"""
    print("\n=== Data Preprocessing ===")
    
    # Display basic information
    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Handle missing values (marked as '?' in this dataset)
    print(f"\nMissing values before cleaning: {df.isin(['?']).sum().sum()}")
    
    # Replace '?' with NaN and then drop rows with missing values
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove any rows that couldn't be converted
    df.dropna(inplace=True)
    
    print(f"Dataset shape after cleaning: {df.shape}")
    
    # Display class distribution
    print(f"\nClass distribution:")
    unique_classes = df['class'].value_counts().sort_index()
    for class_val, count in unique_classes.items():
        print(f"Class {int(class_val)}: {count} samples")
    
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
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        # Feature distributions
        plt.figure(figsize=(15, 12))
        df.hist(figsize=(15, 12), bins=20)
        plt.suptitle('Feature Distributions')
        plt.tight_layout()
        plt.show()
        
        # Class distribution visualization
        plt.figure(figsize=(8, 6))
        df['class'].value_counts().sort_index().plot(kind='bar')
        plt.title('Heart Disease Class Distribution')
        plt.xlabel('Class (0=No Disease, 1-4=Disease Severity)')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.show()
    except Exception as e:
        print(f"Note: Visualization error (continuing without plots): {e}")

def prepare_data_categorical(df):
    """Prepare data for categorical classification (5 classes)"""
    print("\n=== Preparing Data for Categorical Classification ===")
    
    # Separate features and target
    X = df.drop(['class'], axis=1).values
    y = df['class'].values.astype(int)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to categorical for neural network
    y_train_cat = to_categorical(y_train, num_classes=5)
    y_test_cat = to_categorical(y_test, num_classes=5)
    
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    print(f"Categorical target shape: {y_train_cat.shape}")
    
    return X_train_scaled, X_test_scaled, y_train_cat, y_test_cat, y_train, y_test, scaler

def prepare_data_binary(df):
    """Prepare data for binary classification (disease vs no disease)"""
    print("\n=== Preparing Data for Binary Classification ===")
    
    # Separate features and target
    X = df.drop(['class'], axis=1).values
    y = df['class'].values.astype(int)
    
    # Convert to binary: 0 = no disease, 1 = disease (any level)
    y_binary = (y > 0).astype(int)
    
    # Split the data
    X_train, X_test, y_train_bin, y_test_bin = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    print(f"Binary class distribution - No Disease: {(y_train_bin == 0).sum()}, Disease: {(y_train_bin == 1).sum()}")
    
    return X_train_scaled, X_test_scaled, y_train_bin, y_test_bin, scaler

def create_categorical_model(input_dim=13):
    """Create neural network model for categorical classification"""
    model = Sequential([
        Dense(16, input_dim=input_dim, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.3),
        Dense(8, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.3),
        Dense(5, activation='softmax')  # 5 classes (0-4)
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model

def create_binary_model(input_dim=13):
    """Create neural network model for binary classification"""
    model = Sequential([
        Dense(16, input_dim=input_dim, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.3),
        Dense(8, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model

def train_categorical_model(X_train, X_test, y_train_cat, y_test_cat, y_train_orig, y_test_orig):
    """Train and evaluate categorical model"""
    print("\n=== Training Categorical Model (5 Classes) ===")
    
    model = create_categorical_model(input_dim=X_train.shape[1])
    print("Model architecture:")
    print(model.summary())
    
    # Train the model
    history = model.fit(
        X_train, y_train_cat,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )
    
    # Make predictions
    y_pred_cat = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_cat, axis=1)
    
    # Evaluate
    accuracy = accuracy_score(y_test_orig, y_pred_classes)
    print(f"\nCategorical Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_orig, y_pred_classes))
    
    try:
        # Confusion Matrix
        cm = confusion_matrix(y_test_orig, y_pred_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Categorical Model - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    except Exception as e:
        print(f"Note: Visualization error: {e}")
    
    return model, accuracy

def train_binary_model(X_train, X_test, y_train_bin, y_test_bin):
    """Train and evaluate binary model"""
    print("\n=== Training Binary Model (Disease vs No Disease) ===")
    
    model = create_binary_model(input_dim=X_train.shape[1])
    print("Model architecture:")
    print(model.summary())
    
    # Train the model
    history = model.fit(
        X_train, y_train_bin,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )
    
    # Make predictions
    y_pred_prob = model.predict(X_test)
    y_pred_bin = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Evaluate
    accuracy = accuracy_score(y_test_bin, y_pred_bin)
    print(f"\nBinary Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_bin, y_pred_bin))
    
    try:
        # Confusion Matrix
        cm = confusion_matrix(y_test_bin, y_pred_bin)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        plt.title('Binary Model - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    except Exception as e:
        print(f"Note: Visualization error: {e}")
    
    return model, accuracy

def make_prediction(model, scaler, model_type='binary', example_data=None):
    """Make prediction for a new patient"""
    print(f"\n=== Making Predictions ({model_type.title()} Model) ===")
    
    if example_data is None:
        # Example patient data: [age, sex, cp, trestbps, chol, fbs, restecg,
        # thalach, exang, oldpeak, slope, ca, thal]
        example_data = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
    
    # Ensure example_data is the right shape and standardize
    example_array = np.array(example_data).reshape(1, -1)
    example_scaled = scaler.transform(example_array)
    
    # Make prediction
    prediction = model.predict(example_scaled)
    
    print(f"Example patient data: {example_data}")
    
    if model_type == 'binary':
        prob = prediction[0][0]
        pred_class = (prob > 0.5).astype(int)
        print(f"Disease probability: {prob:.4f}")
        print(f"Prediction: {'Heart Disease' if pred_class == 1 else 'No Heart Disease'}")
    else:  # categorical
        probs = prediction[0]
        pred_class = np.argmax(probs)
        print(f"Class probabilities: {probs}")
        print(f"Predicted class: {pred_class}")
        if pred_class == 0:
            print("Prediction: No Heart Disease")
        else:
            print(f"Prediction: Heart Disease (Severity Level {pred_class})")
    
    return prediction

def save_model(model, scaler, model_type):
    """Save the trained model and scaler"""
    project_root = get_project_root()
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(models_dir, f'heart_disease_{model_type}_model.h5')
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save scaler
    import pickle
    scaler_path = os.path.join(models_dir, f'heart_disease_{model_type}_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")

def main():
    """Main function to run the heart disease detection pipeline"""
    print("=== Heart Disease Detection System ===")
    print("Running from heart-disease-diagnosis folder")
    print("=" * 50)
    
    # Download and load data
    df = download_data()
    if df is None:
        return
    
    # Preprocess data
    df_clean = preprocess_data(df)
    
    # Explore data
    explore_data(df_clean)
    
    # Ask user which model type to train
    print("\nWhich model would you like to train?")
    print("1. Binary classification (Disease vs No Disease)")
    print("2. Categorical classification (5 severity levels)")
    print("3. Both models")
    choice = input("Enter your choice (1, 2, or 3): ").strip()
    
    if choice in ['1', '3']:
        # Train binary model
        X_train_bin, X_test_bin, y_train_bin, y_test_bin, scaler_bin = prepare_data_binary(df_clean)
        binary_model, binary_accuracy = train_binary_model(X_train_bin, X_test_bin, y_train_bin, y_test_bin)
        
        # Make example prediction
        make_prediction(binary_model, scaler_bin, 'binary')
        
        # Save model
        save_model(binary_model, scaler_bin, 'binary')
    
    if choice in ['2', '3']:
        # Train categorical model
        X_train_cat, X_test_cat, y_train_cat, y_test_cat, y_train_orig, y_test_orig, scaler_cat = prepare_data_categorical(df_clean)
        categorical_model, categorical_accuracy = train_categorical_model(
            X_train_cat, X_test_cat, y_train_cat, y_test_cat, y_train_orig, y_test_orig
        )
        
        # Make example prediction
        make_prediction(categorical_model, scaler_cat, 'categorical')
        
        # Save model
        save_model(categorical_model, scaler_cat, 'categorical')
    
    if choice == '3':
        print(f"\n=== Model Comparison ===")
        print(f"Binary Model Accuracy: {binary_accuracy:.4f}")
        print(f"Categorical Model Accuracy: {categorical_accuracy:.4f}")
        
        if binary_accuracy > categorical_accuracy:
            print("Binary model performed better - simpler classification is more effective.")
        else:
            print("Categorical model performed better - detailed severity classification is working well.")
    
    print("\n=== Analysis Complete ===")
    print("Check the 'models' folder in your main project directory for saved files.")

if __name__ == "__main__":
    main()