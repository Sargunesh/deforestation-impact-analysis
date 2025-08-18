from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import joblib
import time
import g4f
import pdfkit
from io import BytesIO
import uuid

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(50), nullable=False, default='user')
    predictions = db.relationship('Prediction', backref='user', lazy=True)

# Prediction model to store user predictions
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    input_data = db.Column(db.String(1000), nullable=False)
    prediction = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=db.func.current_timestamp())

# Load ML model and preprocessing objects
loaded_model = joblib.load("best_deforestation_model.pkl")
loaded_label_encoder_country = joblib.load("label_encoder_country.pkl")
loaded_label_encoder_region = joblib.load("label_encoder_region.pkl")
loaded_label_encoder_target = joblib.load("label_encoder_target.pkl")
loaded_scaler = joblib.load("scaler.pkl")

# Function to make predictions
def predict_deforestation(input_data):
    input_data = input_data.copy()
    input_data["Country"] = loaded_label_encoder_country.transform([input_data["Country"]])[0] if input_data["Country"] in loaded_label_encoder_country.classes_ else -1
    input_data["Region"] = loaded_label_encoder_region.transform([input_data["Region"]])[0] if input_data["Region"] in loaded_label_encoder_region.classes_ else -1
    input_df = pd.DataFrame([input_data])
    input_scaled = loaded_scaler.transform(input_df)
    prediction = loaded_model.predict(input_scaled)
    return loaded_label_encoder_target.inverse_transform(prediction)[0]

# Function to generate responses using GPT-4
def generate_response(user_input):
    """Generate a response using GPT-4."""
    try:
        response = g4f.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": user_input}],
            temperature=0.6,
            top_p=0.9
        )
        return response.strip() if response else "Chatbot: Sorry, I didn't understand that."
    except Exception as e:
        return f"Chatbot: Error: {e}"

# Configure wkhtmltopdf path
pdfkit_config = pdfkit.configuration(wkhtmltopdf='C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe')

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = 'user'  # Default role for registration
        
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))

        new_user = User(username=username, password=hashed_password, role=role)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            flash('Login successful!', 'success')
            if user.role == 'admin':
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('user_dashboard'))
        else:
            flash('Invalid credentials!', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('role', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('landing'))

@app.route('/admin_dashboard')
def admin_dashboard():
    if 'user_id' not in session or session.get('role') != 'admin':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('login'))
    
    users = User.query.all()
    return render_template('admin_dashboard.html', users=users)

@app.route('/user_dashboard')
def user_dashboard():
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('login'))
    
    user_predictions = Prediction.query.filter_by(user_id=session['user_id']).all()
    return render_template('user_dashboard.html', predictions=user_predictions)

@app.route('/index', methods=['GET', 'POST'])
def index():
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        input_data = {
            "Year": int(request.form["Year"]),
            "Country": request.form["Country"],
            "Region": request.form["Region"],
            "Forest_Area_km2": float(request.form["Forest_Area_km2"]),
            "Population_Density": float(request.form["Population_Density"]),
            "Agricultural_Land_Percent": float(request.form["Agricultural_Land_Percent"]),
            "Rainfall_mm": float(request.form["Rainfall_mm"]),
            "Temperature_C": float(request.form["Temperature_C"]),
            "Road_Density_km_per_km2": float(request.form["Road_Density_km_per_km2"]),
            "Mining_Activity": float(request.form["Mining_Activity"]),
            "Protected_Area_Percent": float(request.form["Protected_Area_Percent"]),
            "Deforestation_Rate": float(request.form["Deforestation_Rate"]),
            "Fire_Incidents": int(request.form["Fire_Incidents"]),
            "Logging_Activity": float(request.form["Logging_Activity"]),
            "Urban_Expansion_Rate": float(request.form["Urban_Expansion_Rate"]),
            "Carbon_Emissions_kt": float(request.form["Carbon_Emissions_kt"]),
        }
        prediction = predict_deforestation(input_data)
        
        # Save prediction to database
        new_prediction = Prediction(
            user_id=session['user_id'],
            input_data=str(input_data),
            prediction=prediction
        )
        db.session.add(new_prediction)
        db.session.commit()
        
        # Generate text about the project using GPT-4
        project_description = generate_response("Tell me about deforestation prediction and its importance.")
        
        # Store input_data and prediction in session for PDF download
        session['last_input_data'] = input_data
        session['last_prediction'] = prediction
        session['last_project_description'] = project_description
        
        return render_template('result.html', prediction=prediction, project_description=project_description, input_data=input_data)

    return render_template('index.html')

@app.route('/download_pdf')
def download_pdf():
    if 'user_id' not in session or 'last_prediction' not in session:
        flash('No result available to download!', 'warning')
        return redirect(url_for('index'))
    
    # Retrieve data from session
    input_data = session['last_input_data']
    prediction = session['last_prediction']
    project_description = session['last_project_description']
    
    # Render PDF template
    rendered = render_template('pdf_template.html', 
                            prediction=prediction, 
                            project_description=project_description, 
                            input_data=input_data)
    
    # Convert HTML to PDF with configuration
    pdf = pdfkit.from_string(rendered, False, configuration=pdfkit_config)
    
    # Create a BytesIO buffer for the PDF
    pdf_buffer = BytesIO()
    pdf_buffer.write(pdf)
    pdf_buffer.seek(0)
    
    # Send PDF as downloadable file
    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name='deforestation_prediction_result.pdf',
        mimetype='application/pdf'
    )

# Route to create admin (for initial setup, run once manually)
@app.route('/create_admin')
def create_admin():
    admin = User.query.filter_by(role='admin').first()
    if not admin:
        hashed_password = generate_password_hash('admin_password', method='pbkdf2:sha256')
        new_admin = User(username='admin', password=hashed_password, role='admin')
        db.session.add(new_admin)
        db.session.commit()
        flash('Admin created successfully!', 'success')
    else:
        flash('Admin already exists!', 'warning')
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)