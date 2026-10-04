import soundfile as sf
from fastapi import HTTPException
from app.audio_processing import feature_extraction
from io import BytesIO

def calculate_audio_duration(audio_file):
    """Calculate the duration of an audio file in seconds."""
    with sf.SoundFile(audio_file) as audio:
        duration = len(audio) / audio.samplerate

    return duration

def predict_probabilities(model, features):
    """Predict genre probabilities using the loaded model and extracted features."""
    probabilities = model.predict_proba(features)
    return probabilities

def predict_all_audio(model, metadata, audio, n_max_chunks = 10):
    """Predict genre for large audio files by processing in chunks."""
    audio.seek(0)  # Reset the file pointer to the beginning before calculating duration

    audio_duration = calculate_audio_duration(audio)
    audio.seek(0)  # Reset the file pointer to the beginning after calculating duration

    if audio_duration <= 30:
        # If the audio is 30 seconds or less, process it normally
        try:
            features = feature_extraction(audio, metadata)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error during feature extraction: {str(e)}",
            )
        probabilities = predict_probabilities(model, features)
        return probabilities
    else:
        # If the audio is longer than 30 seconds, process it in chunks
        chunk_duration = 30  # seconds
        num_chunks = min(int(audio_duration // chunk_duration), n_max_chunks)

        chunk_probabilities = []

        for i in range(num_chunks):
            audio.seek(0)  # Reset the file pointer to the beginning before reading each chunk
            start_time = i * chunk_duration

            # Extract the chunk from the audio file
        
            with sf.SoundFile(audio) as audio_file:
                sample_rate = audio_file.samplerate
                audio_file.seek(int(start_time * sample_rate))
                chunk_data = audio_file.read(int(chunk_duration * audio_file.samplerate))

            # Create a memory buffer for the chunk
            with BytesIO() as chunk_file:

                sf.write(
                    chunk_file,
                    chunk_data,
                    sample_rate,
                    format="WAV",
                )

                chunk_file.seek(0)
                # Extract features and predict probabilities for the chunk  
                try:
                    features = feature_extraction(chunk_file, metadata)
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Error during feature extraction for chunk {i + 1}: {str(e)}",
                    )
            probabilities = predict_probabilities(model, features)
            chunk_probabilities.append(probabilities)

        # Average the probabilities across all chunks
        averaged_probabilities = sum(chunk_probabilities) / len(chunk_probabilities)
        return averaged_probabilities



