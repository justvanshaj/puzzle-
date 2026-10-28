# Jigsaw Master

Jigsaw Master is a Streamlit application that helps users investigate
where loose jigsaw puzzle pieces may fit.

## Features

- Upload a reference puzzle image.
- Upload a photo of a loose puzzle piece.
- Find possible matches using computer vision.
- Capture photos using the device camera.
- View a highlighted candidate location.

## Requirements

- Python 3.10 or newer
- pip
- A modern web browser

## Installation

Open a terminal in the jigsaw_master directory.

Install the dependencies:

    python -m pip install -r requirements.txt

## Run the application

Execute:

    streamlit run app.py

Streamlit will provide a local web address for opening the app.

## Project structure

    jigsaw_master/
    ├── app.py
    ├── requirements.txt
    ├── manifest.json
    ├── README.md
    ├── icon-192.png
    ├── icon-512.png
    └── icon-maskable-512.png

## App branding

The Streamlit page title and browser tab icon are configured in app.py.

The manifest.json file defines intended PWA metadata, but additional
integration is required before Chrome can use it for installation.
The manifest must be linked from the webpage and served at the
correct address. The icon image files must also exist at those URLs.

## Limitations

The computer-vision matching feature is experimental. A suggested
location is not guaranteed to be correct.
