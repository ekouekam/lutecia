# Libraries to be used ------------------------------------------------------------

import streamlit as st
import requests
import json
import os
from torchmetrics import Metric, WordErrorRate

# from css_tricks import _max_width_

# title and favicon ------------------------------------------------------------

st.set_page_config(page_title="Speech to Text Transcription App", page_icon="")

# _max_width_()

# logo and header -------------------------------------------------

st.text("")
st.image(
    "https://upload.wikimedia.org/wikipedia/fr/0/0d/Logo_OpenClassrooms.png",
    width=125,
)

st.title("Speech to text transcription app")

st.write(
    """
-   Upload a wav file, transcribe it, then export it to a text file!
-   Use cases: call centres, team meetings, training videos, school calls etc.
	    """
)

st.text("")

c1, c2, c3 = st.columns([1, 4, 1])

with c2:

    with st.form(key="my_form"):

        f = st.file_uploader("", type=[".wav"])

        st.info(
            f"""
                    👆 Upload a .wav file. Try a sample: [Sample 01](https://github.com/CharlyWargnier/CSVHub/blob/main/Wave_files_demos/Welcome.wav?raw=true) | [Sample 02](https://github.com/CharlyWargnier/CSVHub/blob/main/Wave_files_demos/The_National_Park.wav?raw=true)
                    """
        )

        submit_button = st.form_submit_button(label="Transcribe")

if f is not None:
    st.audio(f, format="wav")
    path_in = f.name
    # Get file size from buffer
    # Source: https://stackoverflow.com/a/19079887
    old_file_position = f.tell()
    f.seek(0, os.SEEK_END)
    getsize = f.tell()  # os.path.getsize(path_in)
    f.seek(old_file_position, os.SEEK_SET)
    getsize = round((getsize / 1000000), 1)

    if getsize < 5:  # File more than 5 MB
        # To read file as bytes:
        bytes_data = f.getvalue()

        # Load your API key from an environment variable or secret management service
        api_token = st.secrets["API_TOKEN"]

        # endregion API key
        headers = {"Authorization": f"Bearer {api_token}"}
        API_URL = (
            "https://api-inference.huggingface.co/models/facebook/wav2vec2-base-960h"
        )

        def query(data):
            response = requests.request("POST", API_URL, data=data)
            return json.loads(response.content.decode("utf-8"))

        # st.audio(f, format="wav")
        data = query(bytes_data)

        values_view = data.values()
        value_iterator = iter(values_view)
        text_value = next(value_iterator)
        text_value = text_value.lower()

        st.info(text_value)

        c0, c1 = st.columns([2, 2])

        with c0:
            st.download_button(
                "Download the transcription",
                text_value,
                file_name=None,
                mime=None,
                key=None,
                help=None,
                on_click=None,
                args=None,
                kwargs=None,
            )

    else:
        st.warning(
            "🚨 We've limited this demo to 5MB files. Please upload a smaller file."
        )
        st.stop()


else:
    path_in = None
    st.stop()

# graphics dashboard and footer -------------------------------------------------

import pandas as pd

from io import BytesIO
from datasets import load_dataset

# Show transcriptions and comparisons
wer_wav2letter = []
wer_wav2vec = []

# Instantiate the metric
metric = WordErrorRate()

# Load a small sample from the CommonVoice dataset
# Load the CommonVoice dataset from CSV
#local
#url = '/app/common_voice_11_0'
#commonvoice_data = load_dataset(url, 'en', split='train' ,data_files='commonvoice.tsv')
#streaming
commonvoice_data = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train", streaming=True, use_auth_token=st.secrets["API_TOKEN"])

# Selecting the first audio sample
# Get the first sample

first_sample = next(iter(commonvoice_data))

# Get audio URL and transcript from the sample
audio_url = first_sample["path"]
transcription_wav2letter = first_sample["sentence"]

# Download the audio file
audio_response = requests.get(audio_url)

# Check if the request was successful
if audio_response.status_code == 200:
    audio_content = audio_response.content

    # Check audio file size
    audio_size_mb = round(len(audio_content) / (1024 * 1024), 1)  # Size in MB

    if audio_size_mb < 5:  # Proceed if file size is less than 5MB
        # Load your API key from an environment variable or secret management service
        api_token = st.secrets["API_TOKEN"]  # Replace with your API token

        headers = {"Authorization": f"Bearer {api_token}"}
        API_URL = "https://api-inference.huggingface.co/models/facebook/wav2vec2-base-960h"

        def queryhug(data):
            response = requests.post(API_URL, data=data, headers=headers)
            return json.loads(response.content.decode("utf-8"))

        # Convert audio content to BytesIO object
        audio_bytes = BytesIO(audio_content)

        # Perform speech-to-text using the Hugging Face API
        data = queryhug(audio_bytes)

        values_view = data.values()
        value_iterator = iter(values_view)
        text_value = next(value_iterator)
        text_value = text_value.lower()

        print("Transcript:", transcription_wav2letter)
        print("Predicted Text:", text_value)

        # Calculate WER for Wav2Letter++
        wer_wav2letter.append(metric.update(transcription_wav2letter, text_value).compute())

        # Evaluate and compare performance metrics
        average_wer_wav2letter = sum(wer_wav2letter) / len(wer_wav2letter)

        st.write('Average WER for Wav2Letter++:', average_wer_wav2letter)

    else:
        print("Audio file size exceeds 5MB.")
else:
    print("Failed to download audio file.")


# Visualizations or tables showing model performance and comparison
wer_data = {'Model': ['Wav2Letter++', 'Wav2Vec2'], 'WER': [average_wer_wav2letter, average_wer_wav2letter]}
wer_df = pd.DataFrame(wer_data)
