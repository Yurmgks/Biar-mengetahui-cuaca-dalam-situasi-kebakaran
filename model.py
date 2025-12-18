import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

def prepare_data():
    """Mempersiapkan data untuk training"""
    # Load dataset
    df = pd.read_csv('forestfires.csv')
    
    # Encode categorical features
    le_month = LabelEncoder()
    le_day = LabelEncoder()
    
    df['month_encoded'] = le_month.fit_transform(df['month'])
    df['day_encoded'] = le_day.fit_transform(df['day'])
    
    # Features untuk training
    features = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 
                'month_encoded', 'day_encoded']
    
    X = df[features]
    y = df['area']
    
    # Transformasi log pada target (karena data sangat skewed)
    y_log = np.log1p(y)
    
    # Scaling features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_log, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Save model dan preprocessing objects
    joblib.dump(model, 'forestfire_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le_month, 'month_encoder.pkl')
    joblib.dump(le_day, 'day_encoder.pkl')
    
    # Return score
    score = model.score(X_test, y_test)
    return score

def load_model():
    """Memuat model dan preprocessing objects"""
    try:
        model = joblib.load('forestfire_model.pkl')
        scaler = joblib.load('scaler.pkl')
        le_month = joblib.load('month_encoder.pkl')
        le_day = joblib.load('day_encoder.pkl')
        return model, scaler, le_month, le_day
    except:
        # Jika model belum ada, train dulu
        print("Model tidak ditemukan. Melakukan training...")
        score = prepare_data()
        print(f"Model berhasil ditraining dengan R² score: {score:.4f}")
        return load_model()

if __name__ == "__main__":
    # Train model jika file ini dijalankan langsung
    score = prepare_data()
    print(f"Model training selesai! R² Score: {score:.4f}")