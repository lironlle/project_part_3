
from flask import Flask, request, render_template
import pandas as pd
import pickle
from assets_data_prep import prepare_data

app = Flask(__name__)

# Load trained model
with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            # Input mapping with corrected keys
            input_dict = {
                'area': float(request.form.get('area', 0)),
                'room_num': float(request.form.get('room_num', 0)),
                'property_type': request.form.get('property_type', ''),
                'neighborhood': request.form.get('neighborhood', ''),
                'address': request.form.get('address', ''),
                'floor': float(request.form.get('floor', 0)),
                'total_floors': float(request.form.get('total_floors', 0)),
                'garden_area': float(request.form.get('garden_area', 0)),
                'num_of_payments': float(request.form.get('num_of_payments', 0)),
                'days_to_enter': float(request.form.get('days_to_enter', 0)),
                'monthly_arnona': float(request.form.get('monthly_arnona', 0)),
                'building_tax': float(request.form.get('building_tax', 0)),
                'has_parking': int(request.form.get('has_parking', 0)),
                'has_storage': int(request.form.get('has_storage', 0)),
                'elevator': int(request.form.get('elevator', 0)),
                'ac': int(request.form.get('ac', 0)),
                'handicap': int(request.form.get('handicap', 0)),
                'has_bars': int(request.form.get('has_bars', 0)),
                'has_safe_room': int(request.form.get('has_safe_room', 0)),
                'has_balcony': int(request.form.get('has_balcony', 0)),
                'is_furnished': int(request.form.get('is_furnished', 0)),
                'is_renovated': int(request.form.get('is_renovated', 0)),
                'distance_from_center': float(request.form.get('distance_from_center', 1.8))
            }

            print("📥 input_dict:", input_dict)

            df = pd.DataFrame([input_dict])
            df_prepared = prepare_data(df, dataset_type="test")

            print("📊 Prepared DataFrame:")
            print(df_prepared.head())

            prediction = model.predict(df_prepared)[0]

        except Exception as e:
            error = f"שגיאה: {str(e)}"

    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True)
