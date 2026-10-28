import io
import threading
from typing import Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Jigsaw Live Solver",
    page_icon="🧩",
    layout="wide",
)

st.title("🧩 Jigsaw Live Solver")
st.caption("Live camera + AI-assisted visual matching for jigsaw puzzles")

st.info(
    "This application helps you find possible locations for individual "
    "jigsaw pieces. It uses computer vision to compare a piece photograph "
    "with a reference image. Results are suggestions, not guaranteed solutions."
)


# --------------------------------------------------
# CAMERA PROCESSOR
# --------------------------------------------------

class CameraProcessor(VideoProcessorBase):
    """Receive and retain the latest live camera frame."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        with self._lock:
            self._latest_frame = image.copy()

        return frame

    def get_latest_frame(self):
        with self._lock:
            if self._latest_frame is None:
                return None

            return self._latest_frame.copy()


# --------------------------------------------------
# IMAGE UTILITIES
# --------------------------------------------------

def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a PIL image into an OpenCV BGR image."""

    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR image to RGB."""

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_to_width(
    image: np.ndarray,
    max_width: int = 1400,
) -> np.ndarray:
    """Resize large images while preserving their aspect ratio."""

    height, width = image.shape[:2]

    if width <= max_width:
        return image

    scale = max_width / float(width)

    return cv2.resize(
        image,
        (max_width, int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


# --------------------------------------------------
# JIGSAW PIECE MATCHING
# --------------------------------------------------

def orb_match_locations(
    piece_bgr: np.ndarray,
    reference_bgr: np.ndarray,
):
    """
    Compare a puzzle-piece photograph with the reference image.

    Uses ORB keypoints, descriptor matching, and RANSAC homography.

    Returns:
        annotated_reference: Reference image with possible matches.
        status: Human-readable explanation.
        match_count: Number of accepted feature matches.
    """

    piece = resize_to_width(piece_bgr, 700)
    reference = resize_to_width(reference_bgr, 1600)

    piece_gray = cv2.cvtColor(
        piece,
        cv2.COLOR_BGR2GRAY,
    )

    reference_gray = cv2.cvtColor(
        reference,
        cv2.COLOR_BGR2GRAY,
    )

    annotated = reference.copy()

    orb = cv2.ORB_create(
        nfeatures=2500,
        scaleFactor=1.2,
        nlevels=8,
    )

    keypoints_piece, descriptors_piece = (
        orb.detectAndCompute(piece_gray, None)
    )

    keypoints_reference, descriptors_reference = (
        orb.detectAndCompute(reference_gray, None)
    )

    # Check whether both images contain enough visual features.

    if (
        descriptors_piece is None
        or descriptors_reference is None
        or len(keypoints_piece) < 4
        or len(keypoints_reference) < 4
    ):
        return (
            annotated,
            "Not enough distinctive visual features were found. "
            "Try a closer, sharper photograph with less glare.",
            0,
        )

    # Match visual descriptors.

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=False,
    )

    pairs = matcher.knnMatch(
        descriptors_piece,
        descriptors_reference,
        k=2,
    )

    good_matches = []

    for pair in pairs:
        if len(pair) == 2:
            first, second = pair

            if first.distance < 0.72 * second.distance:
                good_matches.append(first)

    # Too few matches usually means that the piece lacks
    # enough distinctive visual details.

    if len(good_matches) < 4:
        return (
            annotated,
            f"Only {len(good_matches)} useful feature matches "
            "were found. Try a piece containing lettering, a "
            "distinctive colour boundary, or a unique pattern.",
            len(good_matches),
        )

    # Prepare corresponding feature coordinates.

    source_points = np.float32(
        [
            keypoints_piece[match.queryIdx].pt
            for match in good_matches
        ]
    ).reshape(-1, 1, 2)

    destination_points = np.float32(
        [
            keypoints_reference[match.trainIdx].pt
            for match in good_matches
        ]
    ).reshape(-1, 1, 2)

    # Estimate the geometric relationship between the images.

    homography, mask = cv2.findHomography(
        source_points,
        destination_points,
        cv2.RANSAC,
        5.0,
    )

    inliers = (
        int(mask.sum())
        if mask is not None
        else 0
    )

    # Draw a possible piece location if a transformation
    # can be estimated.

    if homography is not None and inliers >= 4:

        piece_height, piece_width = piece_gray.shape

        corners = np.float32(
            [
                [0, 0],
                [piece_width, 0],
                [piece_width, piece_height],
                [0, piece_height],
            ]
        ).reshape(-1, 1, 2)

        projected_corners = cv2.perspectiveTransform(
            corners,
            homography,
        )

        polygon = np.int32(projected_corners)

        reference_height, reference_width = reference.shape[:2]

        if (
            np.isfinite(polygon).all()
            and np.max(np.abs(polygon))
            < max(reference_width, reference_height) * 5
        ):

            cv2.polylines(
                annotated,
                [polygon],
                True,
                (0, 0, 255),
                4,
                cv2.LINE_AA,
            )

            center = np.mean(
                polygon.reshape(-1, 2),
                axis=0,
            ).astype(int)

            cv2.circle(
                annotated,
                tuple(center),
                8,
                (0, 0, 255),
                -1,
            )

            status = (
                "Possible matching location highlighted in red. "
                f"Feature matches: {len(good_matches)}. "
                f"Geometric inliers: {inliers}. "
                "Verify the result manually."
            )

            return annotated, status, len(good_matches)

    # If the piece outline cannot be estimated, mark the
    # strongest individual feature matches instead.

    strongest_matches = sorted(
        good_matches,
        key=lambda match: match.distance,
    )[:20]

    for match in strongest_matches:

        x, y = keypoints_reference[
            match.trainIdx
        ].pt

        cv2.circle(
            annotated,
            (int(x), int(y)),
            8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    status = (
        f"Found {len(good_matches)} feature matches, but "
        "could not calculate a reliable piece outline. "
        "Red circles indicate candidate feature locations."
    )

    return annotated, status, len(good_matches)


# --------------------------------------------------
# SIDEBAR: REFERENCE IMAGE
# --------------------------------------------------

with st.sidebar:

    st.header("🖼️ Reference Image")

    reference_upload = st.file_uploader(
        "Upload the completed puzzle picture",
        type=["png", "jpg", "jpeg", "webp"],
        key="reference",
    )

    st.divider()

    st.header("📸 Camera Tips")

    st.markdown(
        """
        - Use bright, indirect lighting.
        - Keep the camera parallel to the puzzle.
        - Avoid reflections and shadows.
        - Photograph one loose piece at a time.
        - Keep the entire printed area of the piece visible.
        """
    )


reference_bgr = None

if reference_upload is not None:

    try:
        reference_pil = Image.open(
            reference_upload
        ).convert("RGB")

        reference_bgr = pil_to_bgr(reference_pil)

    except Exception as exc:
        st.error(
            f"Could not read the reference image: {exc}"
        )


# --------------------------------------------------
# MAIN TABS
# --------------------------------------------------

camera_tab, matching_tab, help_tab = st.tabs(
    [
        "📹 Live Camera",
        "🔎 Match a Piece",
        "ℹ️ Help",
    ]
)


# --------------------------------------------------
# TAB 1: LIVE CAMERA
# --------------------------------------------------

with camera_tab:

    st.subheader("Live Camera Preview")

    st.write(
        "Press START and allow camera permission. "
        "For mobile devices, camera access generally requires HTTPS."
    )

    camera_context = webrtc_streamer(
        key="jigsaw-live-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=CameraProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

    left_column, right_column = st.columns(2)

    with left_column:

        st.markdown("### 📷 Capture a Photograph")

        captured_photo = st.camera_input(
            "Take a photo of a puzzle piece",
            key="still_camera",
        )

        if captured_photo is not None:

            st.session_state[
                "captured_still_bytes"
            ] = captured_photo.getvalue()

            st.success(
                "Photograph captured! Open the Match a Piece tab."
            )

    with right_column:

        st.markdown("### Camera Status")

        if camera_context.state.playing:

            st.success("Live camera is running.")

        else:

            st.warning(
                "Camera is not running. Press START above."
            )

        if camera_context.video_processor:

            latest_frame = (
                camera_context.video_processor.get_latest_frame()
            )

            if latest_frame is not None:

                st.image(
                    bgr_to_rgb(latest_frame),
                    caption="Latest camera frame",
                    use_container_width=True,
                )

            else:

                st.caption(
                    "Waiting for the first camera frame..."
                )


# --------------------------------------------------
# TAB 2: MATCH A PUZZLE PIECE
# --------------------------------------------------

with matching_tab:

    st.subheader("Find a Possible Piece Location")

    if reference_bgr is None:

        st.warning(
            "First upload your puzzle reference image "
            "using the sidebar."
        )

    else:

        st.image(
            bgr_to_rgb(reference_bgr),
            caption="Puzzle reference image",
            use_container_width=True,
        )

    st.markdown(
        """
        Upload a close-up photo of one loose piece.
        The application will compare its visual features
        with the reference picture.
        """
    )

    piece_source = st.radio(
        "Choose the piece photograph source",
        [
            "Upload a photo",
            "Use captured camera photo",
        ],
        horizontal=True,
    )

    piece_bgr = None

    if piece_source == "Upload a photo":

        piece_upload = st.file_uploader(
            "Upload a close-up photograph of one piece",
            type=["png", "jpg", "jpeg", "webp"],
            key="piece_upload",
        )

        if piece_upload is not None:

            try:

                piece_pil = Image.open(
                    piece_upload
                ).convert("RGB")

                piece_bgr = pil_to_bgr(piece_pil)

            except Exception as exc:

                st.error(
                    f"Could not read the piece photograph: {exc}"
                )

    else:

        captured_bytes = st.session_state.get(
            "captured_still_bytes"
        )

        if captured_bytes:

            try:

                captured_pil = Image.open(
                    io.BytesIO(captured_bytes)
                ).convert("RGB")

                piece_bgr = pil_to_bgr(captured_pil)

            except Exception as exc:

                st.error(
                    f"Could not read the captured photograph: {exc}"
                )

        else:

            st.info(
                "Open Live Camera and capture a photograph first."
            )

    if piece_bgr is not None:

        st.image(
            bgr_to_rgb(piece_bgr),
            caption="Piece photograph to analyse",
            use_container_width=True,
        )

    analyse_button = st.button(
        "🔍 Analyse Piece Against Reference",
        type="primary",
        disabled=(
            reference_bgr is None
            or piece_bgr is None
        ),
    )

    if analyse_button:

        with st.spinner(
            "Comparing visual features..."
        ):

            annotated, status, match_count = (
                orb_match_locations(
                    piece_bgr,
                    reference_bgr,
                )
            )

        st.subheader("Matching Result")

        st.write(status)

        st.image(
            bgr_to_rgb(annotated),
            caption="Reference image with candidate clues",
            use_container_width=True,
        )

        if match_count < 4:

            st.warning(
                "A reliable match was not found. Try a "
                "sharper photograph or a piece with distinctive artwork."
            )

        else:

            st.info(
                "The highlighted location is only a candidate. "
                "Check the artwork, orientation, and physical piece shape."
            )


# --------------------------------------------------
# TAB 3: HELP AND LIMITATIONS
# --------------------------------------------------

with help_tab:

    st.subheader("How to Use the App")

    st.markdown(
        """
        **Step 1:** Upload a clear reference image.

        **Step 2:** Start the live camera and allow permission.

        **Step 3:** Photograph one loose puzzle piece or upload its photo.

        **Step 4:** Open Match a Piece.

        **Step 5:** Press Analyse Piece Against Reference.

        **Step 6:** Check the highlighted candidate location manually.
        """
    )

    st.warning(
        """
        Limitations:

        - The current matching algorithm uses ORB feature matching.
        - Plain colours and glossy surfaces can be difficult to match.
        - Different camera angles and piece rotations may reduce accuracy.
        - The app does not automatically solve all 1,000 pieces.
        - A highlighted location is not a guaranteed placement.
        """
    )

    st.markdown(
        """
        **Privacy:** If the app is deployed on a cloud service,
        uploaded and captured images are sent to that service for
        processing. The live camera component streams video frames
        while it is active.
        """
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Jigsaw Live Solver | Streamlit + OpenCV | "
    "Computer-vision puzzle assistance prototype"
)
