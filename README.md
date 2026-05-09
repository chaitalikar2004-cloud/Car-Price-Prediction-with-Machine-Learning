# Car-Price-Prediction-with-Machine-Learning
Car Price Prediction Using Machine Learning

This project is a Machine Learning-based web application that predicts the selling price of used cars based on various features such as present price, fuel type, kilometers driven, transmission type, and ownership details.

The project uses the Random Forest Regressor algorithm for accurate price prediction and is built using Python, Scikit-learn, Pandas, and Flask.

Project Structure
Car-Price-Prediction/
│
├── README.md 
├── car data.csv 
├── car.py
├── car_price_model.pkl
└── requirements.txt
Dataset

Dataset used in this project:

Car Price Prediction (Used Cars) Dataset from Kaggle

Dataset Link:
https://www.kaggle.com/datasets/vijayaadithyanvg/car-price-predictionused-cars

Technologies Used
Python
Pandas
NumPy
Scikit-learn
Flask
Joblib
Features
Data preprocessing
Feature encoding
Machine learning model training
Car price prediction
Model evaluation
Model saving using Joblib
Flask web application
Machine Learning Algorithm

The project uses:

Random Forest Regressor
Installation

Clone the repository:

git clone https://github.com/your-username/Car-Price-Prediction.git

Move into the project directory:

cd Car-Price-Prediction

Install required libraries:

pip install -r requirements.txt
Run the Application

Run the Flask application:

python app.py
Model Training

The machine learning model is trained using:

Train-Test Split
Random Forest Regression
Feature preprocessing and encoding
Future Improvements
Add Streamlit frontend
Deploy on Heroku or Render
Add advanced model tuning
Improve UI design
Add more datasets
Output

The application predicts the estimated selling price of a used car based on user input features.

Author

Chaitali Kar

License

This project is for educational and learning purposes.

Conclusion

This project demonstrates how Machine Learning can be used to predict used car prices accurately using historical data and feature analysis. By applying data preprocessing, feature encoding, and the Random Forest Regressor algorithm, the model is able to provide reliable price predictions.

The project also showcases the integration of Machine Learning with a Flask web application, making the prediction system user-friendly and practical for real-world applications. This project helped in understanding the complete Machine Learning workflow including data handling, model training, evaluation, and deployment.
