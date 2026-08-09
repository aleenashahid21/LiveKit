import librosa
import numpy as np

def extract_features(y, sr=16000):
    """
    Extract MFCCs + spectral centroid + spectral rolloff features
    from an audio signal.
    
    Parameters:
        y (np.ndarray): Audio time series
        sr (int): Sampling rate (default 16kHz)
    
    Returns:
        np.ndarray: Feature vector
    """
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = np.mean(centroid)

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff)

    # Combine into one vector
    features = np.hstack([mfccs_mean, centroid_mean, rolloff_mean])
    return features
