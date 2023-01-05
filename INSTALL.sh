###########################################################
###   
###   ███████╗██╗░░░██╗░█████╗░
###   ██╔════╝██║░░░██║██╔══██╗
###   █████╗░░╚██╗░██╔╝███████║
###   ██╔══╝░░░╚████╔╝░██╔══██║
###   ███████╗░░╚██╔╝░░██║░░██║
###   ╚══════╝░░░╚═╝░░░╚═╝░░╚═╝
###   
###########################################################

# Setup virtual environment
python3 -m venv eva-application-venv
source eva-application-venv/bin/activate

# Install EVA application dependencies
pip install -r requirements.txt

# Refer the custom user-defined function (UDF)
cat toxicity_classifier.py

# Convert Jupyter notebook to README markdown
jupyter nbconvert --execute --to markdown README.ipynb