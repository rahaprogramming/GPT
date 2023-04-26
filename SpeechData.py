import requests
import tarfile
import os

# Download the Common Voice dataset
url = "https://common-voice-data-download.s3.amazonaws.com/cv_corpus_v1.tar.gz"
r = requests.get(url)

# Save the dataset to disk
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)
with open(os.path.join(data_dir, "cv_corpus_v1.tar.gz"), "wb") as f:
    f.write(r.content)

# Extract the dataset
with tarfile.open(os.path.join(data_dir, "cv_corpus_v1.tar.gz"), "r:gz") as f:
    f.extractall(data_dir)
