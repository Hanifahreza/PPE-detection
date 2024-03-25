from super_gradients.training import models
from apd_utils import write_video, convert_video
import torch, PIL, os
import streamlit as st

CLASSES = ['Dust Mask', 'Eye Wear', 'Glove', 'Protective Boots', 'Protective Helmet', 'Safety Vest', 'Shield']
SOURCES = ['Images', 'Videos']

# Setting page layout
st.set_page_config(
    page_title="PPE Object Detection using YOLO-NAS",
    page_icon="👷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("PPE Object Detection using YOLO-NAS")

# Sidebar
st.sidebar.header("YOLO-NAS Model Config")

# Model Options
confidence = float(st.sidebar.slider(
    "Select Model Confidence", 0, 100, 40)) / 100

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio("Select Source", SOURCES)

source_img = None
source_vid = None

#with st.spinner('Downloading model..'):
    #model_url = 'https://drive.google.com/file/d/1XOq3OkpQ3OgibjHmYOCMsQPBtqjdf2i3/view?usp=sharing'
    #download_model(model_url)

model = models.get('yolo_nas_m',
                        num_classes=len(CLASSES),
                        checkpoint_path="./models/ckpt_best_yolonas.pth")

device = 'cuda' if torch.cuda.is_available() else "cpu"
device = 'cpu'

if source_radio == 'Images':
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                st.image('default/default_img.png', caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                        use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)
    with col2:
        if source_img is None:
            st.image('default/default_img_res.png', caption="Detected Objects",
                    use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.to(device).predict(uploaded_image,
                                    conf=confidence)
                st.image(res.draw(), caption='Detected Image',
                        use_column_width=True)

elif source_radio == 'Videos':
    source_vid = st.sidebar.file_uploader(
        "Choose a video ...", type=("mp4", "mov", "webM"))

    col1, col2 = st.columns(2)

    with col1:
        if source_vid is None:
            st.image('default/default_img.png', caption="Default Image",
                    use_column_width=True)
        else:
            try:
                uploaded_video = source_vid.getvalue()
                st.video(uploaded_video)
            except Exception as ex:
                st.error("Error occurred while opening the video.")
                st.error(ex)
    with col2:
        if source_vid is None:
            st.image('default/default_img_res.png', caption="Detected Objects",
                    use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                temp_uploaded_path = write_video(source_vid)
                res = model.to(device).predict(temp_uploaded_path, conf=confidence)

                with st.spinner('Processing video ...'):
                    in_temp_res_path = "./temp/result.mp4"
                    out_temp_res_path = "./temp/result2.mp4"

                    res.save(in_temp_res_path)
                    convert_video(in_temp_res_path, out_temp_res_path)
                st.video(out_temp_res_path)

                os.remove(temp_uploaded_path)
                os.remove(in_temp_res_path)
                os.remove(out_temp_res_path)
else:
    st.error("Please select a valid source type!")
