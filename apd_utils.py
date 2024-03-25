import os
import gdown
import subprocess

def write_video(source_vid):
    os.makedirs('temp', exist_ok=True)
    temp_uploaded_path = f'temp/{source_vid.name}'

    with open(temp_uploaded_path, mode='wb') as temp:
        temp.write(source_vid.read())

    return temp_uploaded_path

def convert_video(in_path, out_path):
    command = [
        'ffmpeg',
        '-i', in_path,
        '-vcodec', 'libx264',
        '-y',
        out_path
    ]
    subprocess.run(command)

def download_model(url):
    os.makedirs('models', exist_ok=True)
    gdown.download(url, 'models/ckpt_best_1.pth', fuzzy=True)

def delete_temp():
    files = os.listdir('./temp')
    
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
