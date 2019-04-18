

This repository contains all the files that were producted during "DeepPhysio" development.



In folder "Application" are collected all the files that have been built for the mobile app. 
"DeepPhysio" app has been design and developed using Appcelerator Titanium framework with "Appcelerator Studio" as IDE.
I recommend the use of these technology in order to further develop "DeepPhysio" project, copying all the files in "Application" folder in the new created project.  
Besides, in order to allow the communication between the app and the server, the appcelerator-module "Titanium Socket.io" was used downloading it from the related GitHub repository.



In folder "DensePose modified scripts" are collected the DensePose's scripts that were modified for "DeepPhysio".
The script "infer_simple.py" has to be inserted in "densepose/tools" folder.
Instead, the script "vis.py" has to be inserted in "densepose/detectron/utils" folder.



In folder "Scripts" are collected all Python scripts that were written for training DeepPhysio's datasets and analyzing their results.
All of these scripts can be started using:


>> python scriptName.py



In folder "Server", it can be found the Python script that starts the execution of DeepPhysio's server in the considered workstation.
The server can be started using:


>> python flaskApp.py


In order to execute correctly the script, the Python module "Flask" and "flask-socketio" have to be downloaded and installed properly via pip.


