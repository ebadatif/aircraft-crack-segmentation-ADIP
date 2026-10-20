import os
from roboflow import Roboflow

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("ebad-atif").project("aircraft-skin-defects-merged-final-b8inl")
version = project.version(1)
dataset = version.download("yolov11")                