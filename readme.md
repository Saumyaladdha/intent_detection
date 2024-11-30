# **Intent Detection with BERT**

## **Overview**

This project focuses on detecting user intents using **BERT (Bidirectional Encoder Representations from Transformers)**. By fine-tuning BERT, the system accurately classifies queries into predefined intents, achieving a validation accuracy of **91%** with high precision and recall for most classes.

The model is trained and evaluated on real-world queries related to mattresses and associated services. This document outlines the project, results, and instructions for reproduction.

---

## **Goals**

1. **Accurate Intent Classification**:  
   * Classify user queries into specific intents like `RETURN_EXCHANGE`, `ORDER_STATUS`, `LEAD_GEN`, and more.  
2. **End-to-End Pipeline**:  
   * Build a pipeline covering data preparation, model training, evaluation, and inference.  
3. **Model Performance**:  
   * Achieve high accuracy, precision, recall, and F1-score across all intent categories.

---

## **Dataset**

The dataset (`sofmattress_train.csv`) contains:

* **Sentences**: User queries (e.g., "How do I return a damaged item?").  
* **Labels**: Corresponding intents (e.g., `RETURN_EXCHANGE`, `DELAY_IN_DELIVERY`).

### **Class Distribution**

The dataset includes 21 intent classes, such as:

* `RETURN_EXCHANGE`  
* `LEAD_GEN`  
* `ORDER_STATUS`  
* `DELAY_IN_DELIVERY`  
* `WARRANTY`  
* ...and more.

Each intent represents a specific user query type, ensuring a diverse and balanced dataset.

---

## **Pipeline**

### **1\. Data Preparation**

1. **Tokenization**:  
   * Used the **BERT Tokenizer** to convert sentences into input IDs and attention masks.  
2. **Label Encoding**:  
   * Encoded intent labels into numeric indices for compatibility with the model.  
3. **Dataset Splitting**:  
   * Split the dataset into:  
     * **80% Training Data**  
     * **20% Validation Data**

### **2\. Model Choice**

* **Why BERT?**  
  * BERT is state-of-the-art for text classification, excelling in contextual understanding.  
  * It processes input as token embeddings with attention mechanisms, ideal for multi-class intent detection.

### **3\. Training**

* **Model**: Fine-tuned **BERT-base-uncased** with a classification head.  
* **Loss Function**: Weighted **CrossEntropyLoss** to handle class imbalance.  
* **Optimizer**: AdamW with a learning rate of `2e-5`.  
* **Scheduler**: StepLR to decay the learning rate every 2 epochs.

### **4\. Evaluation**

* Measured:  
  * **Accuracy**: Overall percentage of correct predictions.  
  * **Precision, Recall, F1-Score**: For detailed class-wise performance.  
  * **Validation Loss**: To monitor model learning.

---

## **Model Performance**

### **Overall Metrics**

* **Accuracy**: 91%  
* **Macro Average**:  
  * **Precision**: 0.91  
  * **Recall**: 0.92  
  * **F1-Score**: 0.89  
* **Weighted Average**:  
  * **Precision**: 0.94  
  * **Recall**: 0.91  
  * **F1-Score**: 0.91

### **Detailed Classification Report**


                      `precision    recall  f1-score   support`

         `RETURN_EXCHANGE       0.83      1.00      0.91         5`  
             `LEAD_GEN          0.80      1.00      0.89         4`  
    `DELAY_IN_DELIVERY        1.00      0.50      0.67         2`  
             `ORDER_STATUS      0.50      1.00      0.67         1`  
                  `COD          1.00      1.00      1.00         2`  
   `ABOUT_SOF_MATTRESS        1.00      0.67      0.80         3`

             `accuracy                           0.91        66`  
            `macro avg       0.91      0.92      0.89        66`  
         `weighted avg       0.94      0.91      0.91        66`

### **Loss Metrics**

* **Training Loss (Final Epoch)**: 0.25  
* **Validation Loss (Final Epoch)**: 0.56

---

## **Inference Results**

Example predictions with the trained model:

| Sentence | Predicted Intent |
| ----- | ----- |
| How do I return a damaged item? | RETURN\_EXCHANGE |
| Why hasn’t my package arrived yet? | DELAY\_IN\_DELIVERY |
| Can I pay for my order upon delivery? | ORDER\_STATUS |
| I'm interested in learning more about your services. | LEAD\_GEN |
| Could you schedule a call with a support agent for tomorrow? | LEAD\_GEN |

### **Observations**

* The model performs well on intents like `RETURN_EXCHANGE` and `LEAD_GEN`.  
* Slight misclassification occurs for `ORDER_STATUS` due to semantic overlap with `DELAY_IN_DELIVERY`.

---

## **Steps to Reproduce**

### **1\. Install Dependencies**

Create a virtual environment and install the required libraries:

 
`python3 -m venv venv`  
``source venv/bin/activate  # Use `venv\Scripts\activate` on Windows``  
`pip install -r requirements.txt`

### **2\. Train the Model**

Run the training script:

`python scripts/train.py`

This will save the trained model and tokenizer in the `bert_intent_model/` directory.

### **3\. Test Predictions**

Run the prediction script with sample queries:

 
`python scripts/predict.py`

---

## **Project Files**

### **1\. Directories**

* **bert\_intent\_model/**: Contains the trained model and tokenizer.  
* **data/**: Contains the dataset file (`sofmattress_train.csv`).  
* **scripts/**: Contains all the scripts.

### **2\. Scripts**

* **train.py**: Preprocesses the data, trains the model, and saves the results.  
* **predict.py**: Loads the trained model and makes predictions on sample queries.  
* **utils.py**: Contains helper functions for tokenization and data processing.

### **3\. Configuration**

* **config.json**: Contains hyperparameters and file paths.

---

## **Improvements**

1. **Augment Data**:  
   * Add more examples for intents like `DELAY_IN_DELIVERY` and `ORDER_STATUS`.  
   * Balance class representation.  
2. **Fine-Tuning**:  
   * Experiment with different hyperparameters (learning rate, batch size).  
   * Use **early stopping** to prevent overfitting.  
3. **Advanced Techniques**:  
   * Implement data augmentation (e.g., paraphrasing queries).  
   * Use dropout and regularization to improve generalization.

---

## **Conclusion**

This project demonstrates the successful application of **BERT** for intent detection, achieving high accuracy and robust predictions. By improving data representation and expanding the dataset, the system can be further enhanced for real-world deployment.


