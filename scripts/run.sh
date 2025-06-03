export OPENAI_API_KEY="API-KEY"

PY="python3"

# GPT_VERSION="o3-mini"
GPT_VERSION="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" # together
# GPT_VERSION="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free" # together



# PAPER_NAME="Transformer"
# PDF_PATH="../examples/Transformer.pdf" # .pdf
# PDF_JSON_PATH="../examples/Transformer.json" # .json
# PDF_JSON_CLEANED_PATH="../examples/Transformer_cleaned.json" # _cleaned.json
# OUTPUT_DIR="../outputs/Transformer"
# OUTPUT_REPO_DIR="../outputs/Transformer_repo"

PAPER_NAME="Malvar"
PDF_PATH="../example/Malvar.pdf" # .pdf
PDF_JSON_PATH="../example/Malvar.json" # .json
PDF_JSON_CLEANED_PATH="../example/Malvar_cleaned.json" # _cleaned.json
OUTPUT_DIR="../outputs/Malvar"
OUTPUT_REPO_DIR="../outputs/Malvar_repo"

mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_REPO_DIR

echo $PAPER_NAME

echo "------- Preprocess -------"

$PY ../codes/0_pdf_process.py \
    --input_json_path ${PDF_JSON_PATH} \
    --output_json_path ${PDF_JSON_CLEANED_PATH} \


echo "------- PaperCoder -------"

$PY ../codes/1_planning.py \
    --paper_name $PAPER_NAME \
    --gpt_version ${GPT_VERSION} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR}


$PY ../codes/1.1_extract_config.py \
    --paper_name $PAPER_NAME \
    --output_dir ${OUTPUT_DIR}

cp -rp ${OUTPUT_DIR}/planning_config.yaml ${OUTPUT_REPO_DIR}/config.yaml

$PY ../codes/2_analyzing.py \
    --paper_name $PAPER_NAME \
    --gpt_version ${GPT_VERSION} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR}

$PY ../codes/3_coding.py  \
    --paper_name $PAPER_NAME \
    --gpt_version ${GPT_VERSION} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --output_repo_dir ${OUTPUT_REPO_DIR} \
