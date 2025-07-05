from flask import Flask, render_template, request, redirect, url_for, flash, abort, jsonify
from datetime import datetime, date, timedelta
import database as db # Use the alias 'db' for clarity
import models # Use the models module for prediction
import logging
import os # For secret key

app = Flask(__name__)
# Secret key: Use environment variable or generate one
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'a_default_dev_secret_key_change_me')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.INFO) # Ensure Flask logger respects level

# --- Database Initialization ---
try:
    db.init_db()
    app.logger.info("Database initialization check complete.")
except Exception as e:
    app.logger.error(f"FATAL: Database initialization failed: {e}", exc_info=True)
    # Consider exiting if DB is critical: raise SystemExit(f"Database initialization failed: {e}")

# --- Context Processors ---
@app.context_processor
def inject_now():
    """Inject current datetime into templates."""
    return {'now': datetime.utcnow()}

# --- Routes ---

@app.route('/')
def index():
    """Renders the home page."""
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register_donor():
    """Handles donor registration (GET shows form, POST processes it)."""
    if request.method == 'POST':
        form_data = request.form.to_dict() # Get form data as dict
        app.logger.debug(f"Received registration data: {form_data}")
        try:
            donor_data = {
                'name': form_data.get('name'),
                'age': int(form_data.get('age')) if form_data.get('age') else None,
                'gender': form_data.get('gender'),
                'blood_type': form_data.get('blood_type'),
                'weight': float(form_data.get('weight')) if form_data.get('weight') else None,
                'hemoglobin_level': float(form_data.get('hemoglobin_level')) if form_data.get('hemoglobin_level') else None,
                'blood_pressure': form_data.get('blood_pressure'),
                'last_donated': form_data.get('last_donated') if form_data.get('last_donated') else None,
                'alcoholic_status': form_data.get('alcoholic_status'),
                # Convert checkboxes/radio 'yes'/'on' to 1, else 0
                'travelled_recently': 1 if form_data.get('travelled_recently') == 'yes' else 0,
                'travel_details': form_data.get('travel_details'),
                'had_recent_illness': 1 if form_data.get('had_recent_illness') == 'yes' else 0,
                'illness_details': form_data.get('illness_details'),
                'recent_tattoo_piercing': 1 if form_data.get('recent_tattoo_piercing') == 'yes' else 0,
                'is_pregnant_breastfeeding': 1 if form_data.get('is_pregnant_breastfeeding') == 'yes' else 0,
                'medical_conditions': form_data.get('medical_conditions'),
                'medications': form_data.get('medications')
            }

            # Basic Server-Side Validation (enhance as needed)
            required_fields = ['name', 'age', 'gender', 'blood_type', 'alcoholic_status']
            missing_fields = [f for f in required_fields if not donor_data.get(f)]
            if missing_fields:
                 flash(f'Missing required fields: {", ".join(missing_fields)}', 'error')
                 # Re-render form with submitted values
                 return render_template('register_donor.html', form_data=form_data)

            if donor_data.get('age') is not None and not (16 <= donor_data['age'] <= 100):
                 flash('Age must be between 16 and 100.', 'error')
                 return render_template('register_donor.html', form_data=form_data)

            # Special validation: If female, check pregnancy/breastfeeding consistency
            if donor_data.get('gender') == 'F' and donor_data.get('is_pregnant_breastfeeding') not in [0, 1]:
                 flash('Please specify pregnancy/breastfeeding status for female donors.', 'warning')
                 # Allow submission but warn, or make it an error:
                 # return render_template('register_donor.html', form_data=form_data)
            elif donor_data.get('gender') != 'F':
                 donor_data['is_pregnant_breastfeeding'] = 0 # Ensure it's 0 if not Female

            # Add donor to database
            donor_id = db.add_donor(donor_data)

            if donor_id:
                flash(f"Donor '{donor_data['name']}' registered successfully!", 'success')
                # Optional: Immediately check eligibility?
                # return redirect(url_for('check_eligibility', donor_id=donor_id))
                return redirect(url_for('list_donors'))
            else:
                flash('Failed to register donor. Database error occurred.', 'error')
                # Keep form data for user convenience
                return render_template('register_donor.html', form_data=form_data)

        except ValueError as e:
            flash(f'Invalid input data type: {e}. Please check numeric fields.', 'error')
            app.logger.warning(f"ValueError during registration: {e} - Form data: {form_data}", exc_info=True)
            return render_template('register_donor.html', form_data=form_data)
        except Exception as e:
            app.logger.error(f"Error during donor registration: {e}", exc_info=True)
            flash('An unexpected error occurred during registration.', 'error')
            return render_template('register_donor.html', form_data=form_data)

    # For GET request, show the empty form
    return render_template('register_donor.html', form_data={}) # Pass empty dict for consistency


@app.route('/donors')
def list_donors():
    """Displays the list of registered donors."""
    try:
        donors_list = db.get_all_donors()
        # Convert Row objects to dictionaries for easier template access if needed,
        # but templates can access Row objects like dicts anyway.
        # donors_list = [dict(row) for row in donors_list]
        return render_template('donors.html', donors=donors_list)
    except Exception as e:
        app.logger.error(f"Error fetching donor list: {e}", exc_info=True)
        flash('Could not retrieve donor list.', 'error')
        return render_template('donors.html', donors=[]) # Show empty list

@app.route('/check_eligibility/<int:donor_id>')
def check_eligibility(donor_id):
    """Fetches donor data and predicts eligibility using the ML model."""
    try:
        donor = db.get_donor_by_id(donor_id)
        if not donor:
            app.logger.warning(f"Eligibility check requested for non-existent donor ID: {donor_id}")
            abort(404) # Donor not found

        donor_dict = dict(donor)
        app.logger.info(f"Checking eligibility for donor: {donor_dict.get('name', 'N/A')} (ID: {donor_id})")

        # Call the prediction function from models.py
        is_eligible, message, details = models.predict_eligibility(donor_dict)

        # Extract potential factors if not eligible
        factors_list = []
        if not is_eligible and details.get('potential_factors'):
            # Format factors nicely
            factors_list = [f"{key}: {value}" for key, value in details['potential_factors'].items()]

        return render_template('eligibility_result.html',
                               donor=donor_dict,
                               is_eligible=is_eligible,
                               message=message, # The main prediction message
                               details=details, # Contains confidence etc.
                               factors=factors_list) # Formatted list of factors

    except Exception as e:
        app.logger.error(f"Error checking eligibility for donor ID {donor_id}: {e}", exc_info=True)
        flash(f'An error occurred while checking eligibility for donor {donor_id}.', 'error')
        return redirect(url_for('list_donors'))

# --- Error Handlers ---

@app.errorhandler(404)
def page_not_found(e):
    """Renders the 404 error page."""
    app.logger.warning(f"404 Not Found: {request.path} - {e}")
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Renders the 500 error page."""
    app.logger.error(f"500 Internal Server Error: {request.path} - {e}", exc_info=True)
    return render_template('500.html'), 500

@app.errorhandler(Exception) # Catch other exceptions
def handle_exception(e):
    """Handles uncaught exceptions."""
    # Handle specific exceptions if needed, otherwise general 500
    if isinstance(e, (KeyError, ValueError)): # Example specific handling
         app.logger.warning(f"Data Error: {request.path} - {e}", exc_info=True)
         flash("There was a problem processing the data.", "error")
         # Redirect to a safe page or render 500
         return render_template('500.html'), 500 # Or redirect(url_for('index'))
    else:
        app.logger.error(f"Unhandled Exception: {request.path} - {e}", exc_info=True)
        return render_template('500.html'), 500


# --- Main Execution ---

if __name__ == '__main__':
    # Ensure model is trained/loaded before starting the app
    try:
        app.logger.info("Loading/Training ML model...")
        models.load_or_train_model() # This will train if model file is missing/invalid
        app.logger.info("ML Model ready.")
    except Exception as e:
         app.logger.error(f"Failed to load/train ML model at startup: {e}", exc_info=True)
         # Decide if the app should run without the model - currently it will try to train on first predict

    # Set debug=False for production
    # Use host='0.0.0.0' to make it accessible on your network
    app.run(debug=True, host='0.0.0.0', port=5000)
    # For production use: waitress-serve --host 0.0.0.0 --port 5000 app:app
    # Or: gunicorn -w 4 -b 0.0.0.0:5000 app:app