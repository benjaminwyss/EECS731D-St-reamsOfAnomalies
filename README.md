# EECS 731 Project 6: D(St)reams of Anomalies
Submission by Benjamin Wyss

## Project Overview

Examining New York City taxi passenger quantities to detect anomalies in the total number of passengers per thirty minute interval.

### Data Set Used

Numenta Anomaly Benchmark (NAB) Datasets - Real Known Cause Anomalies - NYC Taxi Dataset - Taken from: https://github.com/numenta/NAB/tree/master/data on 10/15/20

### Results

Overall, eleven different anomaly detection models examined NYC taxi passenger quantities per 30 minute interval, and the best performing model was a local outlier factor model with passenger quantity as the only input feature. Out of 10,320 data points and 5 anomalies caused by real events, the model detected 1 true positive, 4 false negatives, and 18 false positives--3 of which being close enough in time to other true positive values that the event causing the real anomaly could still be deduced. This model's predictions are accurate enough to determine the cause of 80% of the real anomalies while only predicting anomalies in 0.184% of the dataset (19 predicted anomalies); in total, this is a great result.