guide line for easy setup
1) python 3 
	sudo apt-get install python3-dev libffi-dev libssl-dev -y
	wget https://www.python.org/ftp/python/3.6.3/Python-3.6.3.tar.xz
	tar xJf Python-3.6.3.tar.xz
	cd Python-3.6.3
	./configure
	make
	sudo make install
	sudo pip3 install --upgrade pip
2) install dilb
	
3) opencv3
	https://www.alatortsev.com/2018/09/05/installing-opencv-3-4-3-on-raspberry-pi-3-b/
4) firebase
5) install gps (new 6m)
6) Download shape_predictor_68_face_landmarks.dat
go to cmd find directory and run the below code

python drowsy.py --cascade haarcascade_frontalface_default.xml \
	--shape-predictor shape_predictor_68_face_landmarks.dat