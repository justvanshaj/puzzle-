import io
import threading
from typing import Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Jigsaw Master",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧩 Jigsaw Master")
st.caption("Live camera and computer-vision puzzle assistant")

st.info(
    "Upload a completed puzzle image, photograph a loose piece, "
    "and let the application search for possible visual matches."
)


# ==========================================
# CAMERA PROCESSOR
# ==========================================

class CameraProcessor(VideoProcessorBase):

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


# ==========================================
# IMAGE UTILITIES
# ==========================================

def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_to_width(
    image: np.ndarray,
    max_width: int = 1400,
) -> np.ndarray:

    height, width = image.shape[:2]

    if width <= max_width:
        return image

    scale = max_width / float(width)

    return cv2.resize(
        image,
        (max_width, int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


# ==========================================
# PUZZLE PIECE MATCHING
# ==========================================

def orb_match_locations(
    piece_bgr: np.ndarray,
    reference_bgr: np.ndarray,
):

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

    kp_piece, des_piece = orb.detectAndCompute(
        piece_gray,
        None,
    )

    kp_reference, des_reference = orb.detectAndCompute(
        reference_gray,
        None,
    )

    if (
        des_piece is None
        or des_reference is None
        or len(kp_piece) < 4
        or len(kp_reference) < 4
    ):
        return (
            annotated,
            "Not enough distinctive visual features. "
            "Try a clearer, closer photograph.",
            0,
        )

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=False,
    )

    pairs = matcher.knnMatch(
        des_piece,
        des_reference,
        k=2,
    )

    good_matches = []

    for pair in pairs:

        if len(pair) == 2:

            first, second = pair

            if first.distance < 0.72 * second.distance:
                good_matches.append(first)

    if len(good_matches) < 4:

        return (
            annotated,
            f"Only {len(good_matches)} useful feature matches. "
            "Try a piece with a distinctive pattern or lettering.",
            len(good_matches),
        )

    source_points = np.float32([
        kp_piece[m.queryIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    destination_points = np.float32([
        kp_reference[m.trainIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(
        source_points,
        destination_points,
        cv2.RANSAC,
        5.0,
    )

    inliers = int(mask.sum()) if mask is not None else 0

    if homography is not None and inliers >= 4:

        height, width = piece_gray.shape

        corners = np.float32([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ]).reshape(-1, 1, 2)

        projected = cv2.perspectiveTransform(
            corners,
            homography,
        )

        polygon = np.int32(projected)

        ref_height, ref_width = reference.shape[:2]

        if (
            np.isfinite(polygon).all()
            and np.max(np.abs(polygon))
            < max(ref_width, ref_height) * 5
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

            return (
                annotated,
                f"Possible location marked in red. "
                f"Matches: {len(good_matches)}; "
                f"geometric inliers: {inliers}.",
                len(good_matches),
            )

    for match in sorted(
        good_matches,
        key=lambda m: m.distance,
    )[:20]:

        x, y = kp_reference[match.trainIdx].pt

        cv2.circle(
            annotated,
            (int(x), int(y)),
            8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return (
        annotated,
        "Found visual matches, but could not estimate a "
        "reliable piece outline. Red circles are candidate clues.",
        len(good_matches),
    )


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("🖼️ Puzzle Reference")

    reference_upload = st.file_uploader(
        "Upload completed puzzle image",
        type=["png", "jpg", "jpeg", "webp"],
        key="reference",
    )

    st.divider()

    st.markdown(
        """
        ### Camera tips

        - Use bright, indirect lighting.
        - Avoid glare and shadows.
        - Photograph one piece at a time.
        - Keep the camera parallel to the table.
        """
    )


reference_bgr = None

if reference_upload is not None:

    try:
        reference_image = Image.open(
            reference_upload
        ).convert("RGB")

        reference_bgr = pil_to_bgr(reference_image)

    except Exception as exc:
        st.error(f"Could not read reference image: {exc}")


# ==========================================
# APPLICATION TABS
# ==========================================

camera_tab, match_tab, help_tab = st.tabs([
    "📹 Live Camera",
    "🔎 Match a Piece",
    "ℹ️ Help",
])


# ==========================================
# LIVE CAMERA
# ==========================================

with camera_tab:

    st.subheader("Live Camera")

    st.write(
        "Press START and allow camera permission. "
        "Mobile camera access generally requires HTTPS."
    )

    camera_context = webrtc_streamer(
        key="jigsaw-master-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=CameraProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

    left, right = st.columns(2)

    with left:

        st.subheader("📷 Capture a Photograph")

        captured_photo = st.camera_input(
            "Photograph a puzzle piece",
            key="still_camera",
        )

        if captured_photo is not None:

            st.session_state[
                "captured_still_bytes"
            ] = captured_photo.getvalue()

            st.success(
                "Photo captured. Open Match a Piece."
            )

    with right:

        st.subheader("Camera Status")

        if camera_context.state.playing:
            st.success("Camera is running.")
        else:
            st.warning("Press START to activate the camera.")

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


# ==========================================
# MATCH A PIECE
# ==========================================

with match_tab:

    st.subheader("Find a Possible Piece Location")

    if reference_bgr is None:

        st.warning(
            "Upload your completed puzzle reference "
            "image in the sidebar."
        )

    else:

        st.image(
            bgr_to_rgb(reference_bgr),
            caption="Reference image",
            use_container_width=True,
        )

    source = st.radio(
        "Piece photograph source",
        [
            "Upload a photo",
            "Use captured camera photo",
        ],
        horizontal=True,
    )

    piece_bgr = None

    if source == "Upload a photo":

        piece_upload = st.file_uploader(
            "Upload close-up of one piece",
            type=["png", "jpg", "jpeg", "webp"],
            key="piece_upload",
        )

        if piece_upload is not None:

            try:

                piece_image = Image.open(
                    piece_upload
                ).convert("RGB")

                piece_bgr = pil_to_bgr(piece_image)

            except Exception as exc:

                st.error(f"Could not read piece image: {exc}")

    else:

        captured = st.session_state.get(
            "captured_still_bytes"
        )

        if captured:

            piece_image = Image.open(
                io.BytesIO(captured)
            ).convert("RGB")

            piece_bgr = pil_to_bgr(piece_image)

        else:

            st.info(
                "Capture a photograph in the Live Camera tab first."
            )

    if piece_bgr is not None:

        st.image(
            bgr_to_rgb(piece_bgr),
            caption="Selected puzzle piece",
            use_container_width=True,
        )

    if st.button(
        "🔍 Analyse Piece",
        type="primary",
        disabled=(
            reference_bgr is None
            or piece_bgr is None
        ),
    ):

        with st.spinner("Analysing visual features..."):

            annotated, message, count = orb_match_locations(
                piece_bgr,
                reference_bgr,
            )

        st.subheader("Matching Result")

        st.write(message)

        st.image(
            bgr_to_rgb(annotated),
            caption="Reference with candidate clues",
            use_container_width=True,
        )

        if count < 4:

            st.warning(
                "No reliable match found. Try a sharper "
                "photo or a piece with a distinctive pattern."
            )

        else:

            st.info(
                "The highlighted location is a candidate, "
                "not a guaranteed placement. Verify the physical fit."
            )


# ==========================================
# HELP
# ==========================================

with help_tab:

    st.subheader("How to Use Jigsaw Master")

    st.markdown(
        """
        1. Upload a reference image.
        2. Start the live camera.
        3. Capture or upload one loose puzzle piece.
        4. Open Match a Piece.
        5. Analyse the piece.
        6. Verify the suggested location manually.
        """
    )

    st.warning(
        "This is a prototype. Visual matching can fail on plain colours, "
        "reflections, blurry photos, and rotated pieces. It does not "
        "automatically solve the entire puzzle."
    )

    st.markdown(
        "**Privacy:** On a cloud-hosted app, uploaded images are sent "
        "to the hosting service for processing. Camera frames stream "
        "while the camera is active."
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Jigsaw Master | Streamlit + OpenCV"
)
