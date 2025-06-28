# conda create -n hetddi python=3.9 -y
# conda activate hetddi
python3.10 -m venv hetddi
source hetddi/bin/activate

pip install --upgrade pip
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117
pip install dgl==1.1.1 -f https://data.dgl.ai/wheels/cu117/repo.html

pip install gdown
gdown --folder https://drive.google.com/drive/folders/1VKbVVzAcv_e3UgxId-Jrpac2SKqnCWeN
mv HetDDI/* . && rmdir HetDDI

echo $LD_LIBRARY_PATH
#>> /usr/local/nvidia/lib:/usr/local/nvidia/lib64

(sudo) find / -name "libcusparse.so.11" 2>/dev/null
#>> /opt/conda/lib/python3.11/site-packages/nvidia/cusparse/lib/libcusparse.so.11

export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH

echo 'export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

echo $LD_LIBRARY_PATH
# 출력에 /opt/conda/lib/python3.11/site-packages/nvidia/cusparse/lib 포함돼 있으면 성공
#>> /opt/conda/lib/python3.11/site-packages/nvidia/cusparse/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# conda install -c rdkit rdkit=2022.09.1 -y
pip install rdkit-pypi
python -c "from rdkit import Chem; print(Chem.MolFromSmiles('CCO'))"
#>> <rdkit.Chem.rdchem.Mol object at 0x7fd32c8af1b0>

conda install -c dglteam dgllife -y

pip uninstall numpy -y
pip install "numpy<2"

pip install easydict

nohup python main.py > log/main.log 2>&1 &