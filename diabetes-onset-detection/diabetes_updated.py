# Diabetes Onset Detection using neural network and grid search
# Updated for Python 3 and modern TensorFlow/Keras
# Designed to work in diabetes-onset-detection subfolder

import pandas as pd
import numpy as np
import requests
import warnings
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Add parent directory to path to access shared resources
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def get_project_root():
    """Get the project root directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)

def download_data():
    """Download and load the Pima Indians Diabetes dataset"""
    project_root = get_project_root()
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    data_file = os.path.join(data_dir, 'pima-indians-diabetes.data')
    
    try:
        # Check if file already exists
        if os.path.exists(data_file):
            print("Dataset already exists, loading from file...")
        else:
            # Updated URL for the dataset
            url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
            
            print("Downloading dataset...")
            response = requests.get(url)
            response.raise_for_status()
            
            # Save to local file
            with open(data_file, 'w') as f:
                f.write(response.text)
            print("Dataset downloaded successfully!")
        
        # Column names for the dataset
        names = ['n_pregnant', 'glucose_concentration', 'blood_pressure_mm_hg', 
                'skin_thickness_mm', 'serum_insulin_mu_u_ml', 'BMI', 
                'pedigree_function', 'age', 'class']
        
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
    """Preprocess the diabetes dataset"""
    print("\n=== Exploratory Data Analysis ===")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create DataFrame for easier plotting
        feature_names = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 
                        'insulin', 'bmi', 'pedigree', 'age']
        df_viz = pd.DataFrame(X, columns=feature_names)
        df_viz['diabetes'] = y
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Feature distributions
        plt.subplot(2, 3, 1)
        df_viz.groupby('diabetes')['glucose'].hist(alpha=0.7, bins=20)
        plt.xlabel('Glucose Level')
        plt.ylabel('Frequency')
        plt.title('Glucose Distribution by Diabetes Status')
        plt.legend(['No Diabetes', 'Diabetes'])
        
        plt.subplot(2, 3, 2)
        df_viz.groupby('diabetes')['bmi'].hist(alpha=0.7, bins=20)
        plt.xlabel('BMI')
        plt.ylabel('Frequency')
        plt.title('BMI Distribution by Diabetes Status')
        plt.legend(['No Diabetes', 'Diabetes'])
        
        plt.subplot(2, 3, 3)
        df_viz['diabetes'].value_counts().plot(kind='bar')
        plt.title('Diabetes Class Distribution')
        plt.xlabel('Class (0=No Diabetes, 1=Diabetes)')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        
        plt.subplot(2, 3, 4)
        df_viz.groupby('diabetes')['age'].hist(alpha=0.7, bins=15)
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.title('Age Distribution by Diabetes Status')
        plt.legend(['No Diabetes', 'Diabetes'])
        
        plt.subplot(2, 3, 5)
        df_viz.groupby('diabetes')['pregnancies'].hist(alpha=0.7, bins=10)
        plt.xlabel('Number of Pregnancies')
        plt.ylabel('Frequency')
        plt.title('Pregnancies Distribution by Diabetes Status')
        plt.legend(['No Diabetes', 'Diabetes'])
        
        plt.subplot(2, 3, 6)
        # Correlation with target
        correlations = df_viz.corr()['diabetes'].drop('diabetes').sort_values(key=abs, ascending=False)
        correlations.plot(kind='bar')
        plt.title('Feature Correlation with Diabetes')
        plt.ylabel('Correlation')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_viz.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Note: Visualization error (continuing without plots): {e}")
    """Preprocess the diabetes dataset"""
    print("\n=== Data Preprocessing ===")
    
    # Display basic information
    print("Dataset shape:", df.shape)
    print("\nDataset description:")
    print(df.describe())
    
    # Identify zero values that should be NaN (medical impossibilities)
    columns_with_zeros = ['glucose_concentration', 'blood_pressure_mm_hg', 
                         'skin_thickness_mm', 'serum_insulin_mu_u_ml', 'BMI']
    
    # Replace zero values with NaN for medical impossibilities
    for col in columns_with_zeros:
        df[col].replace(0, np.nan, inplace=True)
    
    print(f"\nRows with missing values: {df.isnull().sum().sum()}")
    
    # Drop rows with missing values
    df_clean = df.dropna()
    print(f"Rows after removing missing values: {len(df_clean)}")
    
    # Split features and target
    X = df_clean.iloc[:, 0:8].values
    y = df_clean.iloc[:, 8].values.astype(int)
    
    # Standardize features
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    
    print(f"Final dataset shape: X={X_standardized.shape}, y={y.shape}")
    
    return X_standardized, y, scaler

def explore_data(X, y):
    """Perform exploratory data analysis with visualizations"""
    print("\n=== Exploratory Data Analysis ===")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create DataFrame for easier plotting
        feature_names = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 
                        'insulin', 'bmi', 'pedigree', 'age']
        df_viz = pd.DataFrame(X, columns=feature_names)
        df_viz['diabetes'] = y
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Feature distributions
        plt.subplot(2, 3, 1)
        df_viz.groupby('diabetes')['glucose'].hist(alpha=0.7, bins=20)
        plt.xlabel('Glucose Level')
        plt.ylabel('Frequency')
        plt.title('Glucose Distribution by Diabetes Status')
        plt.legend(['No Diabetes', 'Diabetes'])
        
        plt.subplot(2, 3, 2)
        df_viz.groupby('diabetes')['bmi'].hist(alpha=0.7, bins=20)
        plt.xlabel('BMI')
        plt.ylabel('Frequency')
        plt.title('BMI Distribution by Diabetes Status')
        plt.legend(['No Diabetes', 'Diabetes'])
        
        plt.subplot(2, 3, 3)
        df_viz['diabetes'].value_counts().plot(kind='bar')
        plt.title('Diabetes Class Distribution')
        plt.xlabel('Class (0=No Diabetes, 1=Diabetes)')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        
        plt.subplot(2, 3, 4)
        df_viz.groupby('diabetes')['age'].hist(alpha=0.7, bins=15)
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.title('Age Distribution by Diabetes Status')
        plt.legend(['No Diabetes', 'Diabetes'])
        
        plt.subplot(2, 3, 5)
        df_viz.groupby('diabetes')['pregnancies'].hist(alpha=0.7, bins=10)
        plt.xlabel('Number of Pregnancies')
        plt.ylabel('Frequency')
        plt.title('Pregnancies Distribution by Diabetes Status')
        plt.legend(['No Diabetes', 'Diabetes'])
        
        plt.subplot(2, 3, 6)
        # Correlation with target
        correlations = df_viz.corr()['diabetes'].drop('diabetes').sort_values(key=abs, ascending=False)
        correlations.plot(kind='bar')
        plt.title('Feature Correlation with Diabetes')
        plt.ylabel('Correlation')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_viz.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Note: Visualization error (continuing without plots): {e}")

def create_model(neurons1=8, neurons2=4, dropout_rate=0.0, learning_rate=0.001, 
                activation='relu', kernel_init='uniform'):
    """Create a neural network model for diabetes prediction"""
    model = Sequential([
        Dense(neurons1, input_dim=8, kernel_initializer=kernel_init, activation=activation),
        Dropout(dropout_rate),
        Dense(neurons2, kernel_initializer=kernel_init, activation=activation),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

def optimize_hyperparameters(X, y):
    """Perform grid search to find optimal hyperparameters"""
    print("\n=== Hyperparameter Optimization ===")
    print("This will take several minutes...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create KerasClassifier wrapper
    model = KerasClassifier(
        model=create_model,
        epochs=50,
        batch_size=32,
        verbose=0,
        random_state=42
    )
    
    # Define parameter grid for grid search
    param_grid = {
        'model__neurons1': [8, 16],
        'model__neurons2': [4, 8],
        'model__dropout_rate': [0.0, 0.2],
        'model__learning_rate': [0.001, 0.01],
        'batch_size': [16, 32],
        'epochs': [50, 100]
    }
    
    print("Performing grid search...")
    
    # Perform grid search
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,  # Reduced for faster execution
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    
    grid_results = grid.fit(X, y)
    
    # Display results
    print(f"\nBest accuracy: {grid_results.best_score_:.4f}")
    print(f"Best parameters: {grid_results.best_params_}")
    
    return grid_results

def train_final_model(X, y, best_params=None):
    """Train the final model with best parameters"""
    print("\n=== Training Final Model ===")
    
    if best_params is None:
        # Default parameters if grid search wasn't performed
        model_params = {
            'neurons1': 8,
            'neurons2': 4,
            'dropout_rate': 0.2,
            'learning_rate': 0.001
        }
        epochs = 100
        batch_size = 32
    else:
        # Extract model parameters
        model_params = {k.replace('model__', ''): v for k, v in best_params.items() 
                       if k.startswith('model__')}
        epochs = best_params.get('epochs', 100)
        batch_size = best_params.get('batch_size', 32)
    
    # Create and train final model
    model = create_model(**model_params)
    
    print("Training final model...")
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    
    # Plot training history
    try:
        import matplotlib.pyplot as plt
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
    y_pred = (model.predict(X) > 0.5).astype(int)
    
    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)
    print(f"\nFinal model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    # Show confusion matrix
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Diabetes', 'Diabetes'],
                   yticklabels=['No Diabetes', 'Diabetes'])
        plt.title('Diabetes Detection - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    except Exception as e:
        print(f"Note: Visualization error: {e}")
    
    return model

def make_prediction(model, scaler, example_data=None):
    """Make prediction for a new patient"""
    print("\n=== Making Predictions ===")
    
    if example_data is None:
        # Example patient data [pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age]
        example_data = [1, 85, 66, 29, 0, 26.6, 0.351, 31]
    
    # Ensure example_data is the right shape and standardize
    example_array = np.array(example_data).reshape(1, -1)
    example_standardized = scaler.transform(example_array)
    
    # Make prediction
    prediction_prob = model.predict(example_standardized)[0][0]
    prediction = (prediction_prob > 0.5).astype(int)
    
    print(f"Example patient data: {example_data}")
    print(f"Diabetes probability: {prediction_prob:.4f}")
    print(f"Prediction: {'Diabetes Risk' if prediction == 1 else 'No Diabetes Risk'}")
    
    return prediction

def save_model(model, scaler):
    """Save the trained model and scaler"""
    project_root = get_project_root()
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(models_dir, 'diabetes_detection_model.h5')
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save scaler
    import pickle
    scaler_path = os.path.join(models_dir, 'diabetes_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")

def main():
    """Main function to run the diabetes detection pipeline"""
    print("=== Diabetes Onset Detection System ===")
    print("Running from diabetes-onset-detection folder")
    print("=" * 50)
    
    # Download and load data
    df = download_data()
    if df is None:
        return
    
    # Preprocess data
    X, y, scaler = preprocess_data(df)
    
    # Explore data with visualizations
    explore_data(X, y)
    
    # Ask user if they want to perform hyperparameter optimization
    print("\nWould you like to perform hyperparameter optimization?")
    print("Note: This will take several minutes but may improve accuracy.")
    choice = input("Enter 'y' for yes, 'n' for no (default): ").lower().strip()
    
    best_params = None
    if choice == 'y':
        # Perform hyperparameter optimization
        grid_results = optimize_hyperparameters(X, y)
        best_params = grid_results.best_params_
    
    # Train final model
    model = train_final_model(X, y, best_params)
    
    # Make example prediction
    make_prediction(model, scaler)
    
    # Save model
    save_model(model, scaler)
    
    print("\n=== Analysis Complete ===")
    print("Check the 'models' folder in your main project directory for saved files.")

if __name__ == "__main__":
    main()