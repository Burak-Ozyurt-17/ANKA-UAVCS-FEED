ANKA FEED PROJECT

This is the data backend for the project ANKA-UAVCS. This platform provides a simple set of data and includes a Yolov8 Image Recognition Model. With a keyboard flight simulation

Instructions:

Install all the pip packages with simply executing the install.sh file.

You can then start the flask server with:
sudo -E python3 flask-feed.py

Note that the keyboard library for python requires sudo and won't work with python3 flask-feed.py 

The route /camera is the camera feed from the machines webcam and the /data is simply for the data in JSON format.

These routes can then be used in the main project as a data feed


