# Install requirements.txt
pip install -r requirements.txt

# Install BLANC
git clone https://github.com/PrimerAI/blanc.git
cd blanc
pip install .
cd ..

# Download punkt
python punkt_download.py

# Download MACSum repo
git clone https://github.com/LucaDeGrandis/MACSum.git
