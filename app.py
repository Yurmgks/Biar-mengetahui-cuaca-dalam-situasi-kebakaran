from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from model import load_model
import requests
from flask import request

app = Flask(__name__)

# Load model saat aplikasi dimulai
model, scaler, le_month, le_day = load_model()

# Mapping bulan
month_names = {
    'jan': 'Januari', 'feb': 'Februari', 'mar': 'Maret', 'apr': 'April',
    'may': 'Mei', 'jun': 'Juni', 'jul': 'Juli', 'aug': 'Agustus',
    'sep': 'September', 'oct': 'Oktober', 'nov': 'November', 'dec': 'Desember'
}

# Mapping hari
day_names = {
    'mon': 'Senin', 'tue': 'Selasa', 'wed': 'Rabu', 'thu': 'Kamis',
    'fri': 'Jumat', 'sat': 'Sabtu', 'sun': 'Minggu'
}

def predict_fire_area(features):
    """Memprediksi luas area kebakaran"""
    try:
        # Transformasi input
        features_scaled = scaler.transform([features])
        
        # Prediksi
        y_pred_log = model.predict(features_scaled)
        
        # Transformasi balik dari log
        y_pred = np.expm1(y_pred_log[0])
        
        return max(0, y_pred)  # Tidak boleh negatif
    except Exception as e:
        print(f"Error in prediction: {e}")
        return 0

@app.route('/')
def index():
    """Halaman utama"""
    return render_template('index.html', 
                         month_names=month_names,
                         day_names=day_names)

# ...existing code...
@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint untuk prediksi"""
    try:
        # Ambil data dari form
        data = request.form
        
        # Validasi input
        try:
            ffmc = float(data.get('ffmc', 0))
            dmc = float(data.get('dmc', 0))
            dc = float(data.get('dc', 0))
            isi = float(data.get('isi', 0))
            temp = float(data.get('temp', 0))
            rh = float(data.get('rh', 0))
            wind = float(data.get('wind', 0))
            rain = float(data.get('rain', 0))
            month = data.get('month', 'jan')
            day = data.get('day', 'mon')

            # Optional: lat/lon dari form (kosong jika tidak diisi)
            lat_raw = data.get('lat', '')
            lon_raw = data.get('lon', '')
            lat = float(lat_raw) if lat_raw not in (None, '') else None
            lon = float(lon_raw) if lon_raw not in (None, '') else None
            
            # Encode categorical features
            month_encoded = le_month.transform([month])[0]
            day_encoded = le_day.transform([day])[0]
            
            # Siapkan features array
            features = [
                ffmc, dmc, dc, isi, temp, rh, wind, rain,
                month_encoded, day_encoded
            ]
            
            # Lakukan prediksi
            predicted_area = predict_fire_area(features)
            
            # Tentukan tingkat risiko
            if predicted_area < 1:
                risk_level = "Rendah"
                risk_color = "green"
            elif predicted_area < 10:
                risk_level = "Sedang"
                risk_color = "yellow"
            elif predicted_area < 50:
                risk_level = "Tinggi"
                risk_color = "orange"
            else:
                risk_level = "Sangat Tinggi"
                risk_color = "red"
            
            # Format output (tambahkan lat & lon jika ada)
            result = {
                'success': True,
                'predicted_area': round(predicted_area, 2),
                'risk_level': risk_level,
                'risk_color': risk_color,
                'month': month_names.get(month, month),
                'day': day_names.get(day, day),
                'lat': lat,
                'lon': lon,
                'input_data': {
                    'FFMC': ffmc,
                    'DMC': dmc,
                    'DC': dc,
                    'ISI': isi,
                    'Suhu': f"{temp}°C",
                    'Kelembaban': f"{rh}%",
                    'Angin': f"{wind} km/h",
                    'Hujan': f"{rain} mm"
                }
            }
            
        except ValueError as e:
            result = {
                'success': False,
                'error': 'Input tidak valid. Harap masukkan angka yang benar.'
            }
        except Exception as e:
            result = {
                'success': False,
                'error': f'Terjadi kesalahan: {str(e)}'
            }
            
    except Exception as e:
        result = {
            'success': False,
            'error': f'Terjadi kesalahan sistem: {str(e)}'
        }
    
    return jsonify(result)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint untuk prediksi (JSON input)"""
    try:
        data = request.json
        
        # Validasi dan prediksi
        ffmc = float(data.get('FFMC', 0))
        dmc = float(data.get('DMC', 0))
        dc = float(data.get('DC', 0))
        isi = float(data.get('ISI', 0))
        temp = float(data.get('temp', 0))
        rh = float(data.get('RH', 0))
        wind = float(data.get('wind', 0))
        rain = float(data.get('rain', 0))
        month = data.get('month', 'jan')
        day = data.get('day', 'mon')
        
        # Optional lat/lon
        lat = data.get('lat')
        lon = data.get('lon')
        lat = float(lat) if lat not in (None, '') else None
        lon = float(lon) if lon not in (None, '') else None
        
        # Encode
        month_encoded = le_month.transform([month])[0]
        day_encoded = le_day.transform([day])[0]
        
        features = [
            ffmc, dmc, dc, isi, temp, rh, wind, rain,
            month_encoded, day_encoded
        ]
        
        predicted_area = predict_fire_area(features)
        
        return jsonify({
            'success': True,
            'predicted_area': round(predicted_area, 2),
            'unit': 'hektar',
            'lat': lat,
            'lon': lon
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)