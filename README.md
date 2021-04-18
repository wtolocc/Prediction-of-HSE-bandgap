# Prediction-of-HSE-bandgap
Using the chemical formula and/or PBE bandgap, HSE bandgap can be predicted with different accuracy.

# Requirements
joblib 0.16.0
scikit-learn 0.23.2
matminer 0.6.4
pandas 1.0.5
monty 3.0.2

# Usage: 
Run the main.py. Then follow the prompts to enter the parameters(dimension, formula, or/and PBE bandgap). if the model cannot be load due to different scikit-learn  versions, please run initialize_model.py.
predicted_materials_project_database contains more than 60,000 predicted HSE bandgaps values.
