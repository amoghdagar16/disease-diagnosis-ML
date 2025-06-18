# Autism Spectrum Disorder Screening using Neural Networks
# Updated for Python 3 and modern TensorFlow/Keras
# Designed to work in autism-screening subfolder

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Add parent directory to path to access shared resources
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def get_project_root():
    """Get the project root directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)

def load_sample_data():
    """Create sample autism screening data for demonstration"""
    print("Creating sample autism screening dataset...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate sample data (normally this would be loaded from a CSV file)
    n_samples = 700
    
    # Q-Chat-10 scores (10 questions, each 0 or 1)
    q_scores = np.random.randint(0, 2, size=(n_samples, 10))
    
    # Demographics
    ages = np.random.randint(18, 65, size=n_samples)
    genders = np.random.choice(['Male', 'Female'], size=n_samples)
    ethnicities = np.random.choice(['White', 'Asian', 'Black', 'Hispanic', 'Other'], size=n_samples)
    jaundice = np.random.choice(['yes', 'no'], size=n_samples)
    family_history = np.random.choice(['yes', 'no'], size=n_samples)
    countries = np.random.choice(['United States', 'Canada', 'United Kingdom', 'Australia', 'Other'], size=n_samples)
    used_app_before = np.random.choice(['yes', 'no'], size=n_samples)
    relations = np.random.choice(['Self', 'Parent', 'Relative', 'Healthcare'], size=n_samples)
    
    # Create target variable (ASD classification) based on Q-Chat score with some noise
    q_total_scores = q_scores.sum(axis=1)
    # Higher Q-Chat scores generally indicate higher likelihood of ASD
    asd_probability = (q_total_scores / 10.0) + np.random.normal(0, 0.1, n_samples)
    asd_classes = (asd_probability > 0.6).astype(int)
    
    # Create column names
    q_columns = [f'A{i+1}_Score' for i in range(10)]
    
    # Create DataFrame
    data = pd.DataFrame(q_scores, columns=q_columns)
    data['age'] = ages
    data['gender'] = genders
    data['ethnicity'] = ethnicities
    data['jaundice'] = jaundice
    data['family_history_of_PDD'] = family_history
    data['country_of_res'] = countries
    data['used_app_before'] = used_app_before
    data['relation'] = relations
    data['class'] = asd_classes
    
    print(f"Sample dataset created with {len(data)} records")
    return data

def load_data():
    """Load autism screening data"""
    print("\n=== Loading Autism Screening Data ===")
    print("Running from autism-screening folder")
    
    project_root = get_project_root()
    dataset_path = os.path.join(project_root, 'dataset', 'autism-data.csv')
    
    # Try to load from file first
    try:
        if os.path.exists(dataset_path):
            print(f"Loading autism dataset from: {dataset_path}")
            
            # Define proper column names for autism dataset
            column_names = [
                'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
                'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
                'age', 'gender', 'ethnicity', 'jaundice', 'family_history_of_PDD',
                'country_of_res', 'used_app_before', 'result', 'age_desc', 'relation', 'class'
            ]
            
            # Load CSV without headers and assign proper column names
            df = pd.read_csv(dataset_path, header=None, names=column_names)
            print("Loaded autism dataset from file successfully!")
            
            # Display basic info about the loaded dataset
            print(f"Dataset shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print("\nFirst few rows:")
            print(df.head())
            
            return df
        else:
            print(f"Dataset file not found at: {dataset_path}")
            print("Creating sample data for demonstration...")
            return load_sample_data()
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating sample data for demonstration...")
        return load_sample_data()

def preprocess_data(df):
    """Preprocess the autism screening dataset"""
    print("\n=== Data Preprocessing ===")
    
    # Display basic information
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Handle missing values
    df.dropna(inplace=True)
    print(f"Dataset shape after removing missing values: {df.shape}")
    
    # Handle the class column - convert text to binary
    print(f"\nOriginal class values: {df['class'].unique()}")
    
    # Convert class to binary (1 for ASD, 0 for No ASD)
    # Handle various possible formats
    df['class'] = df['class'].astype(str).str.upper()
    df['class'] = df['class'].map({
        'YES': 1, 'Y': 1, '1': 1, 'TRUE': 1, 'ASD': 1,
        'NO': 0, 'N': 0, '0': 0, 'FALSE': 0, 'NO ASD': 0
    })
    
    # Remove any rows where class mapping failed
    df = df.dropna(subset=['class'])
    df['class'] = df['class'].astype(int)
    
    # Display class distribution
    print(f"\nClass distribution after conversion:")
    class_counts = df['class'].value_counts()
    for class_val, count in class_counts.items():
        class_label = "ASD" if class_val == 1 else "No ASD"
        print(f"{class_label}: {count} samples")
    
    # Remove potentially identifying or result columns
    columns_to_remove = ['result', 'age_desc'] if 'result' in df.columns else []
    if columns_to_remove:
        df = df.drop(columns_to_remove, axis=1)
        print(f"Removed columns: {columns_to_remove}")
    
    return df

def explore_data(df):
    """Perform exploratory data analysis"""
    print("\n=== Exploratory Data Analysis ===")
    
    # Basic statistics
    print("Dataset description:")
    print(df.describe())
    
    # Q-Chat scores analysis
    q_columns = [col for col in df.columns if col.startswith('A') and 'Score' in col]
    
    try:
        if q_columns:
            # Calculate total Q-Chat score
            df['Q_Chat_Total'] = df[q_columns].sum(axis=1)
            
            # Visualize Q-Chat scores by class
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            df.groupby('class')['Q_Chat_Total'].hist(alpha=0.7, bins=10)
            plt.xlabel('Total Q-Chat Score')
            plt.ylabel('Frequency')
            plt.title('Q-Chat Score Distribution by Class')
            plt.legend(['No ASD', 'ASD'])
            
            plt.subplot(2, 2, 2)
            df['class'].value_counts().plot(kind='bar')
            plt.title('Class Distribution')
            plt.xlabel('Class (0=No ASD, 1=ASD)')
            plt.ylabel('Count')
            plt.xticks(rotation=0)
            
            # Age distribution by class
            if 'age' in df.columns:
                plt.subplot(2, 2, 3)
                df.groupby('class')['age'].hist(alpha=0.7, bins=15)
                plt.xlabel('Age')
                plt.ylabel('Frequency')
                plt.title('Age Distribution by Class')
                plt.legend(['No ASD', 'ASD'])
            
            # Gender distribution
            if 'gender' in df.columns:
                plt.subplot(2, 2, 4)
                gender_class = pd.crosstab(df['gender'], df['class'])
                gender_class.plot(kind='bar', stacked=True)
                plt.title('Gender Distribution by Class')
                plt.xlabel('Gender')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                plt.legend(['No ASD', 'ASD'])
            
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"Note: Visualization error (continuing without plots): {e}")

def prepare_features(df):
    """Prepare features and target variables"""
    print("\n=== Feature Preparation ===")
    
    # Separate features and target
    X = df.drop(['class'], axis=1)
    y = df['class']
    
    print(f"Original features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Handle categorical variables with one-hot encoding
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns: {categorical_columns}")
    
    # One-hot encode categorical variables
    X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    
    print(f"Features shape after encoding: {X_encoded.shape}")
    print(f"Feature columns: {list(X_encoded.columns)}")
    
    # Convert to numpy arrays
    X_array = X_encoded.values
    y_array = y.values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_array, y_array, test_size=0.2, random_state=42, stratify=y_array
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_encoded.columns.tolist()

def create_model(input_dim, dropout_rate=0.3, learning_rate=0.001):
    """Create neural network model for autism screening"""
    model = Sequential([
        Dense(32, input_dim=input_dim, activation='relu', kernel_initializer='he_normal'),
        Dropout(dropout_rate),
        Dense(16, activation='relu', kernel_initializer='he_normal'),
        Dropout(dropout_rate),
        Dense(8, activation='relu', kernel_initializer='he_normal'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model

def train_model(X_train, X_test, y_train, y_test):
    """Train and evaluate the autism screening model"""
    print("\n=== Training Autism Screening Model ===")
    
    model = create_model(input_dim=X_train.shape[1])
    print("Model architecture:")
    print(model.summary())
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    try:
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Note: Visualization error: {e}")
    
    # Make predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No ASD', 'ASD']))
    
    try:
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No ASD', 'ASD'],
                   yticklabels=['No ASD', 'ASD'])
        plt.title('Autism Screening - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    except Exception as e:
        print(f"Note: Visualization error: {e}")
    
    return model, accuracy

def make_prediction(model, scaler, feature_names, example_data=None):
    """Make prediction for a new screening"""
    print("\n=== Making Predictions ===")
    
    if example_data is None:
        # Create example data - this should match the feature structure
        print("Using default example data...")
        example_data = np.random.randint(0, 2, size=len(feature_names))
        
        # Set some specific values for demonstration
        if len(example_data) >= 10:
            # Set Q-Chat scores (first 10 features typically)
            example_data[:10] = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]  # Sample Q-Chat responses
    
    # Ensure example_data is the right shape and standardize
    example_array = np.array(example_data).reshape(1, -1)
    example_scaled = scaler.transform(example_array)
    
    # Make prediction
    prediction_prob = model.predict(example_scaled)[0][0]
    prediction = (prediction_prob > 0.5).astype(int)
    
    print(f"Example screening data shape: {example_array.shape}")
    print(f"ASD probability: {prediction_prob:.4f}")
    print(f"Prediction: {'ASD Likely' if prediction == 1 else 'ASD Unlikely'}")
    
    # Provide interpretation
    if prediction_prob > 0.8:
        print("High likelihood - Recommend professional evaluation")
    elif prediction_prob > 0.6:
        print("Moderate likelihood - Consider professional consultation")
    elif prediction_prob > 0.4:
        print("Low to moderate likelihood - Monitor development")
    else:
        print("Low likelihood - Typical development pattern")
    
    return prediction

def save_model(model, scaler, feature_names):
    """Save the trained model, scaler, and feature names"""
    project_root = get_project_root()
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(models_dir, 'autism_screening_model.h5')
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save scaler and feature names
    import pickle
    scaler_path = os.path.join(models_dir, 'autism_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")
    
    features_path = os.path.join(models_dir, 'autism_feature_names.pkl')
    with open(features_path, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"Feature names saved to: {features_path}")

def main():
    """Main function to run the autism screening pipeline"""
    print("=== Autism Spectrum Disorder Screening System ===")
    print("Running from autism-screening folder")
    print("Note: This system is for educational purposes only.")
    print("Professional evaluation is always recommended for diagnosis.")
    print("=" * 60)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Preprocess data
    df_clean = preprocess_data(df)
    if df_clean is None:
        return
    
    # Explore data
    explore_data(df_clean)
    
    # Prepare features
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_features(df_clean)
    
    # Train model
    model, accuracy = train_model(X_train, X_test, y_train, y_test)
    
    # Make example prediction
    make_prediction(model, scaler, feature_names)
    
    # Save model
    save_model(model, scaler, feature_names)
    
    print("\n=== Important Disclaimer ===")
    print("This model is for educational and research purposes only.")
    print("It should NOT be used as a substitute for professional medical diagnosis.")
    print("If you have concerns about autism, please consult a healthcare professional.")
    
    print("\n=== Analysis Complete ===")
    print("Check the 'models' folder in your main project directory for saved files.")

if __name__ == "__main__":
    main()