```shell
conda create -n websailor python=3.11
conda activate websailor
pip install -r requirements.txt

pip install --upgrade pip
pip install uv
uv pip install "sglang" --prerelease=allow

./src/scripts/test.sh
```