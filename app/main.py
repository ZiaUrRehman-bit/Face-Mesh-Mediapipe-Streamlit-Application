import cv2
import time
import mediapipe as mp
import Face_Mesh_Module as fm
import streamlit as st
import numpy as np
import tempfile
from PIL import Image

mpDrawings = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh

DEMO_IMAGE = 'OriginalDemo.jpeg'
DEMO_VIDEO = '1.mp4'

st.title('Face Mesh Application Using Mediapipe')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"]>div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"]>div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Face Mesh Sidebar")
st.sidebar.subheader('parameters')

@st.cache()
def imageResize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    
    if width is None:
        r = width /float(w)
        dim = (int(w*r), height)
    else:
        r = width /float(w)
        dim = (width, int(h*r))

    resizedImage = cv2.resize(image, dim, interpolation=inter)

    return resizedImage

app_mode = st.sidebar.selectbox(
    'Choose the App Mode',
    ['About App', 'Run on Image', 'Run on Video']
)

if app_mode == 'About App':
    st.markdown('In this application we are using **Mediapipe** for creating FaceMesh on Human Faces. **StreamLit** is used to create the Web Grpahical User Interface (GUI).')
    website_url = "https://developers.google.com/mediapipe/solutions/vision/face_landmarker"

    st.write(f"In order to understand & explore more about FaceMesh Visit [{website_url}]({website_url})")
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"]>div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"]>div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    imageURL = "https://developers.google.com/static/ml-kit/vision/face-mesh-detection/images/face_mesh_overview.png"
    st.image(imageURL, caption="Face-Mesh", use_column_width=True)

    st.markdown(
        '''
            # About Me \n
              Hey! this is Engineer **Zia Ur Rehman**.

              If you are interested in building more **AI, Computer Vision and Machine Learning** projects, then visit my GitHub account. You can find a lot of projects with python code.

              - [GitHub](https://github.com/ZiaUrRehman-bit)
              - [LinkedIn](https://www.linkedin.com/in/zia-ur-rehman-217a6212b/) 
              - [Curriculum vitae](https://github.com/ZiaUrRehman-bit/ZiaUrRehman-bit/blob/main/A2023Updated.pdf)
              
              For any query, You can email me.
              *Email:* engrziaurrehman.kicsit@gmail.com
        '''
    )
    GIFImage = 'https://1.bp.blogspot.com/-nUNmjUY0kwo/XIGaggwn58I/AAAAAAAAD4A/tA3S0Tgu5N4rbjlaNEbr_I5GuHbQwHyRgCEwYBhgL/s1600/image6.gif'
    st.sidebar.image(GIFImage, caption="Face-Mesh", use_column_width=True)
elif app_mode == 'Run on Image':

    drawingSpecs = mpDrawings.DrawingSpec(thickness = 1, circle_radius = 1, color=(0, 255, 0))

    st.sidebar.markdown('---')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"]>div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"]>div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**Detected Faces**")
    kpi1_text = st.markdown("0")

    maxFaces = st.sidebar.number_input('Maximum Number of Face', value = 2, min_value=1)
    st.sidebar.markdown('---')
    detectionConf = st.sidebar.slider("Min Detection Confidence", min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')
    
    imgFileBuffer = st.sidebar.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

    if imgFileBuffer is not None:
        image = np.array(Image.open(imgFileBuffer))
    else:
        demoImage = DEMO_IMAGE
        image = np.array(Image.open(demoImage))

    st.sidebar.text("Original Image")
    st.sidebar.image(image)

    faceCount = 0

    with mpFaceMesh.FaceMesh(
    static_image_mode = True,
    max_num_faces = maxFaces,
    min_detection_confidence = detectionConf) as faceMesh:
        
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = faceMesh.process(imageRGB)
        outImage = cv2.cvtColor(imageRGB, cv2.COLOR_RGB2BGR)
        outImage = outImage.copy()

        for faceLandmarks in results.multi_face_landmarks:
            faceCount +=1

            mpDrawings.draw_landmarks(
                image = outImage,
                landmark_list = faceLandmarks,
                connections = mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec = drawingSpecs)
            kpi1_text.write(f"<h1 style = 'text-align: center; color: red;'>{faceCount}</h1>", unsafe_allow_html=True)
        
        st.subheader('Output Image')
        st.image(outImage, use_column_width=True) 

elif app_mode == 'Run on Video':

    st.set_option('deprecation.showfileUploaderEncoding', False)

    useWebcam = st.sidebar.button('Use Webcam')
    # record = st.sidebar.checkbox('Record Video')

    # if record:
    #     st.checkbox("Recording", value=True)  

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"]>div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"]>div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    maxFaces = st.sidebar.number_input('Maximum Number of Face', value = 2, min_value=1)
    st.sidebar.markdown('---')
    detectionConf = st.sidebar.slider("Min Detection Confidence", min_value=0.0, max_value=1.0, value=0.5)
    trackinfConf = st.sidebar.slider("Min Tracking Confidence", min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')

    st.markdown("## Output")
    
    stframe = st.empty
    videoFileBuffer = st.sidebar.file_uploader( "Upload a Video", type=['mp4','mov', 'avi', 'asf', 'm4v'])
    tffile = tempfile.NamedTemporaryFile(delete=False)

    # Get Input video Frames
    if not videoFileBuffer:
        if useWebcam:
            video = cv2.VideoCapture(0)
        else:
            video = cv2.VideoCapture(DEMO_VIDEO)
            tffile = DEMO_VIDEO
    else:
        tffile.write(videoFileBuffer.read())
        video = cv2.VideoCapture(tffile.name)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fpsInput = int(video.get(cv2.CAP_PROP_FPS))



    faceCount = 0

    with mpFaceMesh.FaceMesh(
    static_image_mode = True,
    max_num_faces = maxFaces,
    min_detection_confidence = detectionConf) as faceMesh:
        
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = faceMesh.process(imageRGB)
        outImage = cv2.cvtColor(imageRGB, cv2.COLOR_RGB2BGR)
        outImage = outImage.copy()

        for faceLandmarks in results.multi_face_landmarks:
            faceCount +=1

            mpDrawings.draw_landmarks(
                image = outImage,
                landmark_list = faceLandmarks,
                connections = mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec = drawingSpecs)
            kpi1_text.write(f"<h1 style = 'text-align: center; color: red;'>{faceCount}</h1>", unsafe_allow_html=True)
        
        st.subheader('Output Image')
        st.image(outImage, use_column_width=True)