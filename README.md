# Speech-to-Text LSTM Project

This project implements a Long Short-Term Memory (LSTM) neural network for converting WAV audio files into text labels. The model extracts MFCC (Mel-Frequency Cepstral Coefficients) features from the audio files and uses them for training and prediction.

---

## Project Structure
speech-to-text-lstm/
├── finalproject.py # Main project code
├── data/ # Dataset folder (not uploaded to GitHub)
└── README.md # This README file


- `finalproject.py`: Contains the full Python code for feature extraction, LSTM model creation, training, and prediction.  
- `data/`: Your dataset folder, containing WAV files organized in subfolders named after their labels.  


---

## Requirements

- Python 3.8+  
- Libraries:
  - numpy
  - librosa
  - tensorflow
  - scikit-learn
  - tkinter (for folder selection)

Install required libraries using pip:

```bash
pip install numpy librosa tensorflow scikit-learn
```
##How to Run

1.Execute the main script:

python finalproject.py


2.A folder selection dialog will appear. Choose your dataset folder containing WAV files.

3.The program will:

Extract MFCC features from audio files

Encode labels

Train the LSTM model with early stopping to prevent overfitting

Evaluate the model on a test set

##Notes

The data/ folder is ignored in GitHub to save storage space.

The model uses two LSTM layers with dropout and a fully connected Dense layer for classification.

EarlyStopping callback is used to prevent overfitting during training.

You can customize the number of epochs, batch size, and LSTM units inside finalproject.py.


##License

This project is licensed under the MIT License.
