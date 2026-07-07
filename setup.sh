#Use this tooo install virtual python env and packages
#Ensuuuure to run this from the working directory

echo "=====Creating virtual environment===="
#Install virtual environemnt using existing python install
python3 -m venv "$(pwd)"

echo "=====Calculating Directory for pip3===="

#Simply displaying the binary locatino of pip3
echo "$(pwd)/bin/"

echo "====Installing packages====="

#Installing packages - enter new ones here
"$(pwd)/bin/pip3" install transformers==4.41.2 torch==2.2.2 \
accelerate==0.30.1 numpy==1.26.4 flask flask_cors


echo "====Installation Complete!===="
#Done