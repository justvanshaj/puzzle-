import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Jigsaw Master",
    page_icon="🧩",
    layout="wide",
)

st.title("🧩 Jigsaw Master")
st.caption("Your smart assistant for solving jigsaw puzzles.")

st.info(
    "Upload a picture of your complete puzzle and a photo of a loose piece. "
    "Jigsaw Master will try to identify a possible matching location."
)


def pil_to_cv(image):
    """Convert a PIL image to an OpenCV BGR image."""
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def find_match(reference, piece):
    """Try to locate a puzzle piece using ORB feature matching."""
    reference_cv = pil_to_cv(reference)
    piece_cv = pil_to_cv(piece)

    reference_gray = cv2.cvtColor(reference_cv, cv2.COLOR_BGR2GRAY)
    piece_gray = cv2.cvtColor(piece_cv, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=3000)

    kp_ref, des_ref = orb.detectAndCompute(reference_gray, None)
    kp_piece, des_piece = orb.detectAndCompute(piece_gray, None)

    if des_ref is None or des_piece is None:
        return None, "Not enough visual details found. Try clearer photos."

    if len(kp_ref) < 4 or len(kp_piece) < 4:
        return None, "Not enough features found. Try another photo."

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    pairs = matcher.knnMatch(des_piece, des_ref, k=2)

    good_matches = []

    for pair in pairs:
        if len(pair) == 2:
            first, second = pair

            if first.distance < 0.75 * second.distance:
                good_matches.append(first)

    if len(good_matches) < 8:
        return None, (
            "No reliable match found. Try a closer photo of the piece "
            "with less background."
        )

    source_points = np.float32(
        [kp_piece[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    destination_points = np.float32(
        [kp_ref[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(
        source_points,
        destination_points,
        cv2.RANSAC,
        5.0,
    )

    if homography is None:
        return None, "Could not estimate the location. Try another photo."

    height, width = piece_gray.shape

    corners = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height],
    ]).reshape(-1, 1, 2)

    projected_corners = cv2.perspectiveTransform(
        corners,
        homography,
    )

    result = reference_cv.copy()

    cv2.polylines(
        result,
        [np.int32(projected_corners)],
        True,
        (0, 255, 255),
        4,
    )

    inliers = int(mask.sum()) if mask is not None else 0

    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    result_image = Image.fromarray(result_rgb)

    message = (
        f"Possible match highlighted. "
        f"Matching inliers: {inliers}/{len(good_matches)}. "
        "This is only a suggestion, not a guaranteed placement."
    )

    return result_image, message


tab_match, tab_camera, tab_help = st.tabs([
    "🔎 Match a Piece",
    "📷 Camera",
    "❓ Help",
])


# --------------------------------------------------
# TAB 1: MATCH A PIECE
# --------------------------------------------------

with tab_match:

    col1, col2 = st.columns(2)

    with col1:
        reference_file = st.file_uploader(
            "Upload the complete puzzle/reference image",
            type=["png", "jpg", "jpeg"],
            key="reference_upload",
        )

    with col2:
        piece_file = st.file_uploader(
            "Upload a photo of one loose puzzle piece",
            type=["png", "jpg", "jpeg"],
            key="piece_upload",
        )

    reference_image = None
    piece_image = None

    if reference_file:
        reference_image = Image.open(reference_file).convert("RGB")

        st.subheader("Complete Puzzle Image")
        st.image(
            reference_image,
            use_container_width=True,
        )

    if piece_file:
        piece_image = Image.open(piece_file).convert("RGB")

        st.subheader("Loose Puzzle Piece")
        st.image(
            piece_image,
            width=300,
        )

    if st.button(
        "Find Possible Location",
        type="primary",
        disabled=not (
            reference_image is not None
            and piece_image is not None
        ),
    ):

        with st.spinner("Analyzing puzzle images..."):

            result, message = find_match(
                reference_image,
                piece_image,
            )

        st.write(message)

        if result is not None:
            st.subheader("Suggested Location")

            st.image(
                result,
                caption="Yellow outline indicates a possible match.",
                use_container_width=True,
            )


# --------------------------------------------------
# TAB 2: CAMERA
# --------------------------------------------------

with tab_camera:

    st.subheader("Take a Puzzle Photo")

    st.write(
        "Use your device camera to capture a photo of your puzzle "
        "or a loose piece."
    )

    camera_photo = st.camera_input(
        "Capture a photo",
    )

    if camera_photo:

        st.image(
            camera_photo,
            caption="Captured photo",
            use_container_width=True,
        )

        st.download_button(
            label="Download Photo",
            data=camera_photo.getvalue(),
            file_name="jigsaw_photo.jpg",
            mime="image/jpeg",
        )


# --------------------------------------------------
# TAB 3: HELP
# --------------------------------------------------

with tab_help:

    st.subheader("How to Use Jigsaw Master")

    st.markdown("""
    **Step 1:** Upload a clear image of the complete puzzle artwork.

    **Step 2:** Take a close-up photo of one loose piece.

    **Step 3:** Upload the piece photo.

    **Step 4:** Click **Find Possible Location**.

    **Step 5:** Inspect the highlighted area in the reference image.

    ### Tips for better results

    - Use bright, even lighting.
    - Avoid shadows and reflections.
    - Keep the loose piece in focus.
    - Include as little background as possible.
    - Avoid blurry or heavily rotated photos.

    ### Important limitations

    This application uses experimental computer-vision matching.
    It may not find a match for every piece, especially pieces with
    repeated patterns or very little visual detail.

    A highlighted location is only a suggestion and should be
    checked visually.
    """)

st.divider()

st.caption("Jigsaw Master | Puzzle-solving assistant")
