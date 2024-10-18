# Automatic-Text-Summarization

Customer reviews can often be long and descriptive. Too much information can confuse and overwhelm us. Analyzing these reviews manually, as you can imagine, is really time-consuming. So, Summarizing long text is necessary for efficient reading and processing of information. The shorter the text becomes, the easier and faster you can obtain information.

## Attention Mechanism 

![gg](https://github.com/viveksaip/Automatic-Text-Summarization/blob/main/images/att.png?raw=true)


## Stacked LSTM 

![gg](https://github.com/viveksaip/Automatic-Text-Summarization/blob/main/images/slstm.png?raw=true)


## Result 

Review: one best flavored coffees tried usually like flavored coffees one great serve company love
Original summary: delicious 
Predicted summary: great coffee

Review: really like product super easy order online delivered much cheaper buying gas station stocking good long drives 
Original summary: turkey jerky is great
Predicted summary: great product

Review: item arrived sugar free shipped regular version carmel small internal sticker bottle stated sugar free although company label bottle stated 
Original summary: wrong item
Predicted summary: not good


## Detailed Expalanation 

# Text Summarization Model

## Overview
This project implements an automatic text summarization model using deep learning techniques, specifically a sequence-to-sequence architecture enhanced with attention mechanisms. The goal is to generate concise and coherent summaries from lengthy text documents.

## Key Components

### 1. Data Collection
- Utilized a dataset containing reviews with full text and corresponding summaries.
- Preprocessed the data to clean it and remove duplicates and missing values.

### 2. Text Preprocessing
- Implemented a text cleaning function that:
  - Converts text to lowercase.
  - Removes HTML tags using BeautifulSoup.
  - Expands contractions (e.g., "can't" to "cannot").
  - Removes stop words and non-alphabetic characters.
- Tokenized and padded the cleaned text to create uniform input sequences.

### 3. Model Architecture
- Built a sequence-to-sequence model comprising:
  - **Encoder**: Processes the input text and encodes it into a set of states.
  - **Decoder**: Generates the summary from the encoded states using an attention mechanism.
- The **Attention Layer** allows the decoder to focus on relevant parts of the input, enhancing the summary quality.

### 4. Training the Model
- The model was trained on a training set with early stopping to prevent overfitting.
- Used sparse categorical crossentropy loss function suitable for multi-class classification.

### 5. Generating Summaries
- Implemented a prediction function that processes input text and generates a summary iteratively until reaching an end token or a maximum length.

### 6. Evaluation
- While the main focus was on building the model, evaluation metrics like ROUGE can be utilized to assess the quality of generated summaries against reference summaries.

## Key Learnings and Challenges
- **Challenges**: Tuning model parameters and ensuring effective attention mechanism integration.
- **Learnings**: Gained insights into natural language processing, sequence modeling, and the role of attention mechanisms in improving performance.

## Future Improvements
- Explore advanced architectures, such as Transformer models, to further enhance summarization quality.
- Implement user feedback mechanisms for iterative improvement based on real-world usage.



