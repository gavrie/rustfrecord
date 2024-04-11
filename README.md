pip install uv
uv venv
source .venv/bin/activate

uv pip compile
uv pip compile requirements.in -o requirements.txt
uv pip install -r requirements.txt

maturin develop

export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.9/site-packages/torch/lib
python main.py