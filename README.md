# Touch Analytics

## Overview

**Touch Analytics** is a demo project developed for a class, designed to explore user authentication using AI based on swipe behavior. 
The project collects swipe data from users and trains a model to predict whether new swipes belong to the same user.

> **Note:** In practice, this method is not highly reliable for authentication, as a single user’s swipe patterns can vary significantly, and increasing the number of users increases the chances of overlap.

Despite these limitations, the project serves as a practical showcase for **basic AI and machine learning skills**, including:

- Data preprocessing and cleaning
- Feature scaling using standardization
- Training a supervised classification model with appropriate inputs and outputs

---

## Credit

This project was completed as part of a class. **Only the backend code** was implemented by me. The Android app frontend was provided by the class instructor and is included for demonstration purposes.

---

## Setup

### 1. Database

This project uses **Firebase Realtime Database**. To run the project, you need:

- A Firebase database URL configured in the backend code
- A `firebase_service_key.json` file exported from your Firebase project

> **Important:** User IDs in Firebase must **not be the integer `1`**. If `1` is used as a key, Firebase interprets it as a list instead of a dictionary, which will break the backend code. Use other integers (e.g., `2`, `123`) or string-based IDs.

---

### 2. Android App Frontend

An Android app is included as a `.zip` file. When opened in **Android Studio**, it provides the frontend for the project. This frontend allows users to:

- Log in with a user ID
- Perform swipe actions
- Track swipe data associated with their login

To use this, simply unzip the file and open the file using android studio.
---

### 3. Backend

The backend is responsible for training the AI model. To run it:

1. Update the **Firebase URL** and **service key** in the backend configuration
2. activate the python venv provided in the "favenv" folder with this command for linux users:
`source favenv/bin/activate`
2. Start the backend server
3. The backend will preprocess swipe data, train the model, and provide an endpoint for user authentication

#### Note: the flask routing for this project is not ideal for real training and would be done differently if the directions did not specify exactly 50 swipes and pre-set routing.

---

## Data Storage

This program stores the model as well as the scaler using 'mlp/joblib' and 'scaler.joblib' respecitvely. A file called `user.pkl` is used to store the set of known users using Python’s pickle module. 
This prevents retraining the model for users already seen, and combined this files allow for saved models and users between server restarts.

## Notes

- This project is primarily an educational demonstration, not a production-ready authentication system.
- All features used in the backend are numeric and require consistent formatting in the Firebase database.
- Ensure that the database structure remains consistent to avoid errors when reading data.
