from data_preprocessing import load_and_preprocess_data
from model import train_logistic_regression, train_decision_tree, train_svm, evaluate_model

def main():
    # Load and preprocess data
    X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess_data()
    
    # Train models
    print("Training Logistic Regression model...")
    logreg_model = train_logistic_regression(X_train_scaled, y_train)
    
    print("Training Decision Tree model...")
    dt_model = train_decision_tree(X_train_scaled, y_train)
    
    print("Training SVM model...")
    svm_model = train_svm(X_train_scaled, y_train)
    
    # Evaluate models
    print("\nEvaluating Logistic Regression:")
    y_pred_logreg = logreg_model.predict(X_test_scaled)
    evaluate_model(y_test, y_pred_logreg)
    
    print("\nEvaluating Decision Tree:")
    y_pred_dt = dt_model.predict(X_test_scaled)
    evaluate_model(y_test, y_pred_dt)
    
    print("\nEvaluating SVM:")
    y_pred_svm = svm_model.predict(X_test_scaled)
    evaluate_model(y_test, y_pred_svm)

if __name__ == "__main__":
    main()
