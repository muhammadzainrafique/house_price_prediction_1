from django.shortcuts import render
import joblib
import numpy as np
import pandas as pd

import os

# Get the directory where this views.py file is located
current_dir = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(current_dir, 'model.pkl'))
scaler = joblib.load(os.path.join(current_dir, 'scaler.pkl'))
le = joblib.load(os.path.join(current_dir, 'label_encoder.pkl'))

def home(request):
    return render(request, 'home.html')

def result(request):
    if request.method == 'POST':
        # Debug: Print all POST data
        print("POST data received:", request.POST)
        
        income = float(request.POST['income'])
        age = float(request.POST['age'])
        rooms = float(request.POST['rooms'])
        bedrooms = float(request.POST['bedrooms'])
        population = float(request.POST['population'])
        address = request.POST.get('address', 'No address provided')  # Use get() with default value
        
        # Debug: Print individual values
        print(f"Income: {income}")
        print(f"Age: {age}")
        print(f"Rooms: {rooms}")
        print(f"Bedrooms: {bedrooms}")
        print(f"Population: {population}")
        print(f"Address: {address}")

        # Encode and scale
        address_enc = le.transform([address])[0] if address in le.classes_ else 0
        user_input = pd.DataFrame([[income, age, rooms, bedrooms, population, address_enc]])
        user_input_scaled = scaler.transform(user_input)

        prediction = model.predict(user_input_scaled)[0]

        context = {
            'price': round(prediction, 2),
            'income': income,
            'age': age,
            'rooms': rooms,
            'bedrooms': bedrooms,
            'population': population,
            'address': address
        }
        
        # Debug: Print context
        print("Context being sent to template:", context)
        
        return render(request, 'result.html', context)
