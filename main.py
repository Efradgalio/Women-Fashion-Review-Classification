import preprocessing
import inference

# Main function
def main():
    # Accept review text input from the user
    print("Please enter the review text: ")
    review_text = input()  # Take input from user
    
    # Preprocess the Data
    tfidf_new_data_preprocessed, new_data_preprocessed  = preprocessing.full_preprocessing(review_text)

    # Predict the Data
    logreg_pred, logreg_proba, bert_all_pred = inference.predict(tfidf_new_data_preprocessed, new_data_preprocessed)

    print('')
    print('Review: {}'.format(review_text))
    print('Logistic Regression Predicted --> label: LABEL_{} and SCORE: {}'.format(logreg_pred, logreg_proba))
    print('BERT Predicted --> label: LABEL_{} and SCORE: {}'.format(bert_all_pred[0]['label'], bert_all_pred[0]['score']))

if __name__ == "__main__":
    main()