🧩 Jigsaw Live Solver

A Streamlit application that combines live camera access with basic computer vision to help identify possible locations for individual jigsaw puzzle pieces.

Features

* Live browser-camera preview.
* Capture puzzle-piece photographs.
* Upload individual puzzle pieces.
* Upload a puzzle reference image.
* Compare images using OpenCV ORB feature matching.
* Highlight possible matching locations.
* Support desktop and mobile browsers.

Requirements

* Python 3.10 or 3.11 recommended.
* Internet connection for installing dependencies.
* A webcam or mobile phone camera.
* A reference image of the completed puzzle.

Project Structure

jigsaw_solver_streamlit/
├── app.py
├── requirements.txt
└── README.md

Installation on Windows

Open PowerShell or Command Prompt inside the project directory.

1. Create a virtual environment

py -3.11 -m venv .venv

2. Activate the environment

.venv\Scripts\Activate.ps1

If PowerShell prevents activation, use Command Prompt and run:

.venv\Scripts\activate.bat

3. Install the dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

4. Run the application

streamlit run app.py

Open the local address displayed in your terminal. It is usually:

http://localhost:8501

Using the Application

1. Upload a clear image of the completed puzzle using the sidebar.
2. Open the Live Camera tab.
3. Press START and grant camera permission.
4. Photograph a single puzzle piece, or upload an existing photo.
5. Open the Match a Piece tab.
6. Select the appropriate photograph source.
7. Press Analyse Piece Against Reference.
8. Review the highlighted candidate locations.

Mobile Camera Access

For convenient mobile access, deploy the application to a hosting provider that supports HTTPS.

Open the HTTPS application address in your mobile browser and grant camera permission.

Camera streaming may fail on certain browsers or networks. If that happens, use the photograph capture control or upload an image instead.

Deployment

To deploy through Streamlit Community Cloud:

1. Create a GitHub repository.
2. Upload app.py, requirements.txt, and README.md.
3. Create a new application through Streamlit Community Cloud.
4. Select the repository.
5. Set the main application file to app.py.
6. Deploy the application.
7. Open the deployed HTTPS address on your phone.

Camera streaming can depend on browser permissions, hosting configuration, and network restrictions.

Limitations

This project is a prototype, not a complete jigsaw puzzle solver.

ORB feature matching works best when the photographed piece contains distinctive artwork, lettering, or visual patterns.

It may struggle with:

* Plain colour regions.
* Glossy surfaces and reflections.
* Small or blurry photographs.
* Pieces photographed at different angles.
* Rotated pieces.
* Repeated patterns.

The application does not yet detect and solve all pieces automatically, reconstruct the complete puzzle grid, or guarantee exact placement.

Privacy

When deployed on a cloud host, images are sent to that host for processing. The live camera component streams frames while active.

Avoid uploading sensitive images to a hosting service unless you are comfortable with its data-handling practices.

Technologies

* Python
* Streamlit
* OpenCV
* NumPy
* Pillow
* streamlit-webrtc

License

You may modify this prototype for personal use. Add an explicit open-source license if you plan to publish or distribute it.
