# Melody Generation using LSTM and Bi-LSTM Neural Networks

This project implements and compares two deep learning approaches (LSTM and Bi-LSTM) for generating musical melodies. The models can generate unique melodies while maintaining musical coherence.

## Overview

The project uses two different architectures:
- Long Short-Term Memory (LSTM) Networks
- Bidirectional LSTM (Bi-LSTM) Networks

![Screenshot from 2024-11-26 17-12-38](https://github.com/user-attachments/assets/8f4d3d5e-a7dd-482d-ba51-eadfc34c2243)

Both models are trained on preprocessed musical data and compared using various metrics including training performance and melody uniqueness analysis using Dynamic Time Warping (DTW).


## Requirements

1. Python 3.8 or higher
2. Required libraries:
   ```bash
   pip install tensorflow music21 numpy matplotlib fastdtw scipy
   ```

## Dataset
I am using the publicly available ESAC Dataset (http://esac-data.org/). It has a collection of various folk songs from various continents. I have selected the German Folk Music dataset called "altdeu2" which has 316 music files.


## Preprocessing
Create an empty folder "dataset" in the root directory. The below command generates files (encoded songs) inside the dataset folder and mapping.json (mapping between musical symbols and integers) and file_dataset (single file containing all the encoded songs) in the root directory

```bash
python3 preprocess.py
```
## Training
These commands generate lstm_model.h5 and bilstm_model.h5 files respectively.
```bash
python3 train_lstm.py
```
```bash
python3 train_bilstm.py
```
After training, the below command is used to generate the audio file. You can try out giving any "seed" value from the files generated in "dataset" folder
```bash
python3 melody_generator.py
```
<b>NOTE</b>: You can skip the steps above and run this single script ``` "python3 compare_model.py"``` to train the models, generate the audio files(.mid) and graph making comparison between the models  

These are the generated .mid files for the seed value "69 _ _ _ 69 _ _ _ _ _ 68 _ 69 _ _ _ 71 _ _ _ 72 _ _ _ _ _ 72"
<br>Bi-LSTM - https://drive.google.com/file/d/1ldqko21Ix8hAQavbt_YJjuY9mVMqajW7/view?usp=sharing
<br>LSTM - https://drive.google.com/file/d/1mQ18QLLKWMRHugUNAPmCl1d5-ygWi33K/view?usp=sharing


## Results
![Screenshot from 2024-11-25 10-27-53](https://github.com/user-attachments/assets/1d854a0e-42c8-48bd-a847-9a5ea51fd777)

<br>LSTM Test Loss: 0.3315
<br>Bi-LSTM Test Loss: 0.4439

<br>LSTM Test Accuracy: 0.8945
<br>Bi-LSTM Test Accuracy: 0.9123

<br>Uniqueness(LSTM) : 48.4%
<br>Uniqueness(Bi-LSTM): 65.4%

The result shows a contrasting behaviour i.e Bi-LSTM loss is higher than LSTM but its accuracy is also higher than LSTM. It can happen due big errors in some of the data leading to high loss.
FOr both of the models I have used "Categorical Cross Entropy" loss function and it is sensitive to large errors. Hence ,it can lead to a significant increase in the total loss, even if the majority of predictions are correct.

