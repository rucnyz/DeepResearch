export SEARCH_API_URL="your_search_api_url"
export JINA_API_KEY=jina_b76d29ac793941369e1e0cb6c8d78213PGYCi9gRj0CfegLFyXdGSay9wBnK
export GOOGLE_SEARCH_KEY=475e6b446236268cd0fb9aa42a85b43cb94f2972

export SUMMARY_MODEL_PATH="Qwen/Qwen2.5-32B-Instruct"
export MAX_LENGTH=$((1024 * 31 - 500))

cd src || exit

# The arguments are the model path, the dataset name, and the location of the prediction file.
# bash run.sh <model_path> <dataset> <output_path>

# Dataset names (strictly match the following names):
# - gaia
# - browsecomp_zh (Full set, 289 Cases)
# - browsecomp_en (Full set, 1266 Cases)
# - xbench-deepsearch

bash run.sh Alibaba-NLP/WebSailor-3B gaia outputs