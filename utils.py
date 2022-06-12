import subprocess
import os
import zipfile
import urllib.request as urllib
import shutil
import yaml
from PIL import Image


def train_model(model_name: str, url: str = '', batch_size: int = 5, epochs: int = 15, download_dataset: bool = True,
                use_gpu: bool = False):
    """
    Downloads the given model and Trains it with yolov5 based on the user set batch size and epochs

    :param url: URL to the roboflow dataset formatted for yolov5, if no url given it will use the dataset at model_name
    :param model_name: Name of the model you would like to use
    :param batch_size: Batch size of images for training, change depending on how much ram you have
    :param epochs: Number of iterations of the training set
    :param download_dataset: Whether the program should download the linked model or not
    :param use_gpu: ONLY USE THIS IF YOU HAVE AN NVIDIA GPU WITH CUDATOOLKIT INSTALLED
    :return:
    """

    # Hacky way to find the cudatoolkit version to attempt auto installing the correct pytorch, this can and probably
    #   will break.
    if use_gpu:
        # Decode the binary string of the console output
        nvcc = subprocess.check_output('nvcc --version').decode("utf-8")
        # Very bad way to get the position of the version
        temp_pos = nvcc.find("release ")+8
        temp_substring = nvcc[temp_pos::]
        # Formats the version for the requirements.txt file
        cuda_version = temp_substring[0:temp_substring.find(',')].replace('.', '')

        package_names = ['torch==1.10.0+cu', 'torchvision==0.11.1+cu']

        # Opens requirements.txt to replace the cuda versions
        with open('yolov5/requirements.txt', 'r') as file:
            file_data = file.read()

        print(file_data)
        for package in package_names:
            pack_loc = file_data.find(package)
            file_data = file_data.replace(f'{file_data[pack_loc:pack_loc+len(package)+3]}', f'{package}{cuda_version}')

        with open('yolov5/requirements.txt', 'w') as file:
            file.write(file_data)

    if download_dataset:
        try:
            # If the dataset directory already exists, delete it
            if os.path.isdir(model_name):
                print("Cleaning Old Datasets...")
                shutil.rmtree(model_name)
            # Download and unzip the new dataset
            print("Downloading Roboflow Dataset...")
            path, _ = urllib.urlretrieve(url)
            # Unzip the file
            with zipfile.ZipFile(path, "r") as file:
                file.extractall(model_name)
        except Exception as e:
            print(e)

    try:
        # Edit the yaml file to provide the correct paths to the datasets
        with open(f'{model_name}/data.yaml') as f:
            data = yaml.safe_load(f)
            data['train'] = f'../{model_name}/train/images'
            data['val'] = f'../{model_name}/valid/images'

            with open(f'{model_name}/data.yaml', 'w') as f:
                yaml.safe_dump(data, f, default_flow_style=False)

        # Install or validate required packages for yolov5
        subprocess.run('pip3 install -r requirements.txt', cwd=f'{os.getcwd()}/yolov5', shell=True)

        if use_gpu:
            device = '0'
        else:
            device = 'cpu'

        # Train the model
        subprocess.run(
            f'python ../yolov5/train.py --img 640 --batch {batch_size} --epochs {epochs} --data data.yaml --weights '
            f'yolov5s.pt --device {device}',
            cwd=f'{os.getcwd()}/{model_name}',
            shell=True)

        dir = 'yolov5/runs/train'
        img = Image.open(f"{dir}/{os.listdir(dir)[-1]}/results.png")
        img.show()
    except Exception as e:
        print(e)

def export_model(model_name: str):
    """
    Exports the model as [model_name].blob in this directory
    :param model_name: Name of the model you would like to use
    :return:
    """
    export = input("Do you want to export this model? [Y/n]")
    if export.lower() == 'y':
        print("Exporting model...")

        # Have to delete old blob or else blobconverter refuses to download a new model
        old_model_path = [file for file in os.listdir(os.getcwd()) if file.endswith('.blob')][0]
        os.remove(old_model_path)

        # Export model as onnx, install/validate blobconverter, convert onnx model
        subprocess.run(
            [f'python yolov5/export.py --weights yolov5s.pt --include onnx',
             'pip install blobconverter',
             f'python -m blobconverter --onnx-model yolov5s.onnx --output-dir {os.getcwd()} --shaves 6'],
            cwd=f'{os.getcwd()}',
            shell=True)

        # Change the name of the blob to your selected file name
        model_path = [file for file in os.listdir(os.getcwd()) if file.endswith('.blob')][0]
        os.rename(model_path, f'{model_name}.blob')
    else:
        print("Model export canceled.")

