import streamlit as st
import pickle

# Load the trained model
def load_model(model_path):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Predict income based on user inputs
def predict_income(model, user_inputs, numerical_features):
    # Initialize the feature vector with zeros
    features = [0] * 99  # Set the length to 99 based on the provided column names

    # Create a dictionary to map the selected categorical values to binary-encoded features
    categorical_mapping = {feature: 1 for feature in user_inputs if feature in categorical_features}

    # Update the features with binary-encoded values
    for idx, feature in enumerate(categorical_features):
        if feature in categorical_mapping:
            features[idx] = 1

    # Add the numerical features
    features[-6:] = numerical_features

    if len (features) != 99:
        raise ValueError("Incorrect number of features in the input vector")

    income_prediction = model.predict([features])
    return income_prediction[0]

# Create Streamlit UI to collect user inputs
def main():
    st.title('Income Prediction App')
    st.write('Enter the following information to predict income category:')

    # Load the model
    model_path = ""C:/Users/hp/Downloads/main_pkl.py""   # Replace with the actual path to your model
    model = load_model(model_path)

    # Create dropdown menus for categorical variables
    st.subheader("Categorical Features:")
    user_inputs = []
    for group, features in categorical_groups.items():
        feature = st.selectbox(f"{group}:", features)
        user_inputs.append(f"{group}_{feature}")

    # Numerical Features
    st.subheader("Numerical Features:")
    age = st.slider('Age', 17, 90, 35)
    fnlwgt = st.slider('Final Weight', 100, 1000000, 50000)
    education_num = st.slider('Education Num', 1, 16, 9)
    capital_gain = st.slider('Capital Gain', 0, 99999, 0)
    capital_loss = st.slider('Capital Loss', 0, 4356, 0)
    hours_per_week = st.slider('Hours per Week', 1, 99, 40)

    if st.button('Predict'):
        income_prediction = predict_income(model, user_inputs, [age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week])
        st.write(f'Predicted Income Category: {income_prediction}')

if __name__ == "__main__":
    categorical_groups = {
        "workclass": ['Federal-gov', 'Local-gov', 'Never-worked', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay'],
        "education": ['10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Preschool', 'Prof-school'],
        "marital.status": ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Widowed'],
        "occupation": ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving'],
        "relationship": ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife'],
        "race": ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'White'],
        "native.country": ['Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece', 'Guatemala', 'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico', 'Nicaragua', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan', 'Thailand', 'Trinadad&Tobago', 'United-States', 'Vietnam', 'Yugoslavia', 'others']
    }

    categorical_features = [
        f"{group}_{feature}" for group, features in categorical_groups.items() for feature in features
    ]

    main()
