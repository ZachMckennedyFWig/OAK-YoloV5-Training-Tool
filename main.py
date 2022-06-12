from utils import *



if __name__ == "__main__":
    url = "[PASTE ROBOFLOW LINK HERE]"
    model_name = "[ENTER MODEL NAME HERE]" # Whatever you want the model to be called

    batch_size = 5
    epochs = 10

    # If not already installed, installs yolov5
    if not os.path.isdir('yolov5'):
        subprocess.run('git clone https://github.com/ultralytics/yolov5', cwd=f'{os.getcwd()}/yolov5', shell=True)

    # Trains the model in YOLOv5 for you, use_gpu parameter should only be used if you have an nvidia graphics card
    #   and cudatoolkit installed. It will try and find your version of nvidiatoolkit and use that to get the right
    #   build of PyTorch, if you are getting issues with that it might be better to either just train with your CPU
    #   or move to a google collab notebook.
    train_model(model_name=model_name, url=url, batch_size=batch_size, epochs=epochs,
                use_gpu=True, download_dataset=False)

    # Exports the model as [model_name].blob in this directory.
    export_model(model_name=model_name)














