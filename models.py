import os
import datetime
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer # Keep importer here
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = "blood_donation_model_enhanced.pkl" # New name

# Define features used by the model consistently
# Basic donor info + calculated/binary flags
NUMERIC_FEATURES = ['age', 'weight', 'days_since_last_donation', 'hemoglobin_level',
                  'travelled_recently', 'had_recent_illness', 'recent_tattoo_piercing',
                  'is_pregnant_breastfeeding'] # Binary flags treated as numeric here
CATEGORICAL_FEATURES = ['blood_type', 'gender', 'alcoholic_status']
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES # Order matters for ColumnTransformer

def create_preprocessor():
    """Creates the preprocessor pipeline for the model."""
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), # Handles missing categories
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse=False easier for inspection
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, NUMERIC_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES)
    ], remainder='passthrough') # Keep other columns if any accidentally passed
    return preprocessor

def load_or_train_model():
    """Load existing model or train a new one."""
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading existing model from {MODEL_PATH}...")
        try:
            pipeline = joblib.load(MODEL_PATH)
            if isinstance(pipeline, Pipeline) and 'classifier' in pipeline.named_steps:
                 logger.info("Model loaded successfully.")
                 return pipeline
            else:
                logger.warning("Loaded object is not a valid pipeline. Retraining...")
                return train_model()
        except Exception as e:
            logger.error(f"Error loading model: {e}. Retraining...")
            return train_model()
    else:
        logger.info("No existing model found. Training a new model...")
        return train_model()

def train_model():
    """Train a RandomForest model for blood donation eligibility."""
    try:
        logger.info("Generating synthetic data for training...")
        data = create_sample_data()
        if data.empty:
             logger.error("Failed to generate sample data.")
             return None

        logger.info(f"Generated {len(data)} samples.")
        X = data[ALL_FEATURES]
        y = data['eligible']

        preprocessor = create_preprocessor()

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=120, random_state=42, class_weight='balanced', max_depth=15, min_samples_split=5)) # Tuned slightly
        ])

        logger.info("Splitting data and training model...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        pipeline.fit(X_train, y_train)

        logger.info("Evaluating model...")
        accuracy = pipeline.score(X_test, y_test)
        logger.info(f"Test Accuracy: {accuracy:.4f}")

        try: # Wrap CV in try-except as it can fail with small/imbalanced data sometimes
            cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy') # Use accuracy explicitly
            logger.info(f"Cross-validation Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
        except Exception as cv_e:
             logger.warning(f"Cross-validation failed: {cv_e}")


        y_pred = pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=['Not Eligible', 'Eligible'], zero_division=0)
        logger.info("Classification Report:\n" + report)

        logger.info(f"Saving model to {MODEL_PATH}...")
        joblib.dump(pipeline, MODEL_PATH)
        logger.info("Model training and saving complete.")
        return pipeline

    except ImportError as e:
        logger.error(f"ImportError during training: {e}. Make sure scikit-learn is fully installed.")
        return None
    except Exception as e:
        logger.error(f"Error training model: {e}", exc_info=True)
        return None

def create_sample_data(n_samples=1500):
    """Generate synthetic data including new fields."""
    try:
        np.random.seed(42)
        today = datetime.date.today()

        data = pd.DataFrame({
            'age': np.random.randint(16, 75, n_samples),
            'weight': np.random.normal(70, 15, n_samples),
            'last_donation_date': [today - datetime.timedelta(days=np.random.randint(1, 500)) if np.random.rand() > 0.2 else None for _ in range(n_samples)],
            'hemoglobin_level': np.random.normal(14, 2, n_samples),
            'blood_type': np.random.choice(['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-', 'Unknown', np.nan], n_samples, p=[0.2, 0.05, 0.15, 0.05, 0.05, 0.01, 0.3, 0.09, 0.05, 0.05]),
            'gender': np.random.choice(['M', 'F', 'O', np.nan], n_samples, p=[0.47, 0.47, 0.02, 0.04]),
            'alcoholic_status': np.random.choice(['Never', 'Occasional', 'Regular', 'Recovering', 'Unknown', np.nan], n_samples, p=[0.4, 0.3, 0.1, 0.05, 0.1, 0.05]),
            'travelled_recently': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'had_recent_illness': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'recent_tattoo_piercing': np.random.choice([0, 1], n_samples, p=[0.92, 0.08]),
            'is_pregnant_breastfeeding': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]) # Apply only if gender='F' later
        })

        # Apply pregnancy only to Females
        data.loc[data['gender'] != 'F', 'is_pregnant_breastfeeding'] = 0

        # Calculate days_since_last_donation
        data['days_since_last_donation'] = data['last_donation_date'].apply(
            lambda d: (today - d).days if pd.notnull(d) else 9999 # High value for never/NaN
        )
        data['days_since_last_donation'].fillna(9999, inplace=True)

        # Clean up outliers
        data['weight'] = data['weight'].clip(lower=30, upper=200)
        data['hemoglobin_level'] = data['hemoglobin_level'].clip(lower=5, upper=25)

        # Define eligibility criteria (Simplified - Model learns complex interactions)
        data['eligible'] = (
            data['age'].between(18, 65) &
            (data['weight'] >= 50) &
            (data['days_since_last_donation'] >= 56) &
            (data['travelled_recently'] == 0) & # Simple rule: no recent travel disqualifies (model might learn nuance)
            (data['had_recent_illness'] == 0) & # Simple rule: recent illness disqualifies
            (data['recent_tattoo_piercing'] == 0) & # Simple rule: recent tattoo/piercing disqualifies (often 3-6 months rule)
            (data['is_pregnant_breastfeeding'] == 0) &
            (data['alcoholic_status'] != 'Regular') & # Simple rule: Regular drinkers might be deferred
            (
                ((data['gender'] == 'M') & (data['hemoglobin_level'] >= 13.0)) |
                ((data['gender'] == 'F') & (data['hemoglobin_level'] >= 12.5)) |
                 # Basic check for 'O' or NaN gender/Hgb - assume ineligible if data missing
                (data['gender'].isna() | data['hemoglobin_level'].isna() | (data['gender'] == 'O')) == False
            )
        ).astype(int)

        # Adjust eligibility slightly randomly to make it less deterministic for the model
        random_flip = np.random.rand(n_samples) < 0.05 # Flip 5% of labels
        data['eligible'] = data['eligible'].apply(lambda x: 1-x if np.random.rand() < 0.05 else x)


        # Drop intermediate date column
        data = data.drop(columns=['last_donation_date'])
        # Fill NaNs in features used by model before returning (imputer will handle this in pipeline, but good practice)
        data['weight'].fillna(data['weight'].median(), inplace=True)
        data['hemoglobin_level'].fillna(data['hemoglobin_level'].median(), inplace=True)
        data['blood_type'].fillna('Unknown', inplace=True)
        data['gender'].fillna('O', inplace=True) # Assign NaN gender to 'Other'
        data['alcoholic_status'].fillna('Unknown', inplace=True)


        logger.info(f"Sample Data Head:\n{data[ALL_FEATURES + ['eligible']].head()}")
        logger.info(f"Eligibility distribution:\n{data['eligible'].value_counts(normalize=True)}")


        return data

    except Exception as e:
        logger.error(f"Error creating sample data: {e}", exc_info=True)
        return pd.DataFrame()

def predict_eligibility(donor_data):
    """Predict donor eligibility using the trained model and updated features."""
    model = load_or_train_model()

    if not model:
        logger.error("Model could not be loaded or trained. Cannot predict.")
        return False, "Error: Eligibility model is unavailable.", {}

    try:
        input_dict = {}
        # Map DB fields to model features
        input_dict['age'] = donor_data.get('age')
        input_dict['weight'] = donor_data.get('weight')
        input_dict['hemoglobin_level'] = donor_data.get('hemoglobin_level')
        input_dict['blood_type'] = donor_data.get('blood_type', 'Unknown')
        input_dict['gender'] = donor_data.get('gender', 'O') # Default to 'Other' if missing
        input_dict['alcoholic_status'] = donor_data.get('alcoholic_status', 'Unknown')
        input_dict['travelled_recently'] = int(donor_data.get('travelled_recently', 0)) # Ensure 0/1
        input_dict['had_recent_illness'] = int(donor_data.get('had_recent_illness', 0)) # Ensure 0/1
        input_dict['recent_tattoo_piercing'] = int(donor_data.get('recent_tattoo_piercing', 0)) # Ensure 0/1
        input_dict['is_pregnant_breastfeeding'] = int(donor_data.get('is_pregnant_breastfeeding', 0)) # Ensure 0/1

        # Calculate days_since_last_donation
        last_donated_str = donor_data.get('last_donated')
        days_since = 9999
        if last_donated_str:
            try:
                if isinstance(last_donated_str, str):
                    last_donated_date = datetime.datetime.strptime(last_donated_str, '%Y-%m-%d').date()
                elif isinstance(last_donated_str, datetime.date):
                    last_donated_date = last_donated_str
                else:
                     last_donated_date = None # Treat unexpected types as unknown

                if last_donated_date:
                    days_since = (datetime.date.today() - last_donated_date).days
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not parse last_donated date '{last_donated_str}': {e}. Assuming long time ago.")
                days_since = 9999
        input_dict['days_since_last_donation'] = days_since

        # Ensure all features are present, even if None (imputer handles None)
        for feature in ALL_FEATURES:
            if feature not in input_dict:
                input_dict[feature] = None # Or a suitable default like 0 for numeric binary flags

        # Create DataFrame with columns in the exact order expected by the preprocessor
        input_df = pd.DataFrame([input_dict])[ALL_FEATURES] # Select columns in specific order
        logger.debug(f"Input DataFrame for prediction:\n{input_df.to_string()}")


        # Predict using the pipeline
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        probability_eligible = probabilities[model.classes_.tolist().index(1)] # Probability of the '1' class

        is_eligible = bool(prediction)
        confidence = f"{probability_eligible * 100:.1f}%"
        message = f"Donor is predicted {'ELIGIBLE' if is_eligible else 'NOT ELIGIBLE'}."


        # Provide simple rule-based checks as potential factors
        factors = {}
        if not is_eligible:
            # Basic Checks
            if not (18 <= input_dict.get('age', 0) <= 65): factors['Age'] = f"Outside 18-65 range ({input_dict.get('age')})"
            if not (input_dict.get('weight', 0) >= 50): factors['Weight'] = f"Below 50 kg ({input_dict.get('weight')})"
            if not (input_dict.get('days_since_last_donation', 9999) >= 56): factors['Last Donation'] = f"Less than 56 days ago ({input_dict.get('days_since_last_donation')})"
            # Hemoglobin
            hgb = input_dict.get('hemoglobin_level')
            gender = input_dict.get('gender')
            hgb_check_passed = True
            if hgb is not None and gender and gender != 'O':
                 if gender == 'M' and hgb < 13.0: hgb_check_passed = False
                 if gender == 'F' and hgb < 12.5: hgb_check_passed = False
                 if not hgb_check_passed: factors['Hemoglobin'] = f"Level potentially low ({hgb} g/dL for gender {gender})"
            elif hgb is None:
                factors['Hemoglobin'] = "Level not provided"
            # New Factors
            if input_dict.get('travelled_recently') == 1: factors['Recent Travel'] = "May require deferral (details needed)"
            if input_dict.get('had_recent_illness') == 1: factors['Recent Illness'] = "May require deferral (details needed)"
            if input_dict.get('recent_tattoo_piercing') == 1: factors['Recent Tattoo/Piercing'] = "May require deferral (usually 3-6 months)"
            if input_dict.get('is_pregnant_breastfeeding') == 1: factors['Pregnancy/Breastfeeding'] = "Currently ineligible"
            if input_dict.get('alcoholic_status') == 'Regular': factors['Alcohol Use'] = "Regular use may be a factor"
            # Add checks for medications/medical_conditions if needed (complex)


        details = {
            'model_prediction': is_eligible,
            'confidence_score': probability_eligible,
            'confidence_percent': confidence,
            'potential_factors': factors
        }
        logger.info(f"Prediction for donor {donor_data.get('name', 'N/A')} (ID: {donor_data.get('id', 'N/A')}): {message} (Confidence: {confidence})")

        return is_eligible, message, details

    except Exception as e:
        logger.error(f"Prediction error for donor ID {donor_data.get('id', 'N/A')}: {e}", exc_info=True)
        return False, f"Error during eligibility prediction: {e}", {}

# Example Usage / Training Trigger
if __name__ == "__main__":
    print("Attempting to load or train the enhanced model...")
    model = load_or_train_model()

    if model:
        print("\nEnhanced Model is ready. Testing with example donor...")
        donor_example = {
            'id': 999, 'name': 'Test Donor Eligible', 'age': 35, 'gender': 'F', 'blood_type': 'A+',
            'weight': 60, 'hemoglobin_level': 13.0, 'blood_pressure': '110/70',
            'last_donated': (datetime.date.today() - datetime.timedelta(days=100)).isoformat(),
            'alcoholic_status': 'Occasional', 'travelled_recently': 0, 'travel_details': None,
            'had_recent_illness': 0, 'illness_details': None, 'recent_tattoo_piercing': 0,
            'is_pregnant_breastfeeding': 0, 'medical_conditions': 'None', 'medications': 'None'
        }
        is_eligible, message, details = predict_eligibility(donor_example)
        print(f"\n--- Prediction Result (Eligible Example) ---")
        print(f"Is Eligible: {is_eligible}")
        print(f"Message: {message}")
        print(f"Confidence: {details.get('confidence_percent')}")
        print(f"Potential Factors: {details.get('potential_factors')}")
        print("-------------------------")

        donor_ineligible_example = {
             'id': 998, 'name': 'Test Donor Ineligible', 'age': 28, 'gender': 'M', 'blood_type': 'O-',
            'weight': 75, 'hemoglobin_level': 14.5, 'blood_pressure': '130/85',
            'last_donated': (datetime.date.today() - datetime.timedelta(days=30)).isoformat(), # Too recent
            'alcoholic_status': 'Never', 'travelled_recently': 1, 'travel_details': 'Malaria Zone', # Recent Travel
            'had_recent_illness': 0, 'illness_details': None, 'recent_tattoo_piercing': 1, # Recent Tattoo
            'is_pregnant_breastfeeding': 0, 'medical_conditions': 'None', 'medications': 'None'
        }
        is_eligible, message, details = predict_eligibility(donor_ineligible_example)
        print(f"\n--- Prediction Result (Ineligible Example) ---")
        print(f"Is Eligible: {is_eligible}")
        print(f"Message: {message}")
        print(f"Confidence: {details.get('confidence_percent')}")
        print(f"Potential Factors: {details.get('potential_factors')}")
        print("-------------------------")
    else:
        print("\nEnhanced Model could not be loaded or trained. Please check logs.")