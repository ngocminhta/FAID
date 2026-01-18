MODEL_PATH="" # checkpoint of the trained model
TEST_DATA_PATH="" # path for the data of My Anh https://github.com/myanhtrannguyen/data_authscan/tree/main 
MODEL_NAME="" # hf id of the model
LANGUAGE="" # language of the data "vi" or "en"
DATABASE_PATH="" # path to the saved database
FULL_LABEL=0 # 0 for 3 labels, 1 for 9 labels
K=50

python test_monolingual.py \
    --database_path ${DATABASE_PATH} \
    --test_dataset_path ${TEST_DATA_PATH} \
    --model_name ${MODEL_NAME} \
    --model_path ${MODEL_PATH} \
    --full_labels ${FULL_LABEL} \
    --language ${LANGUAGE} \
    --max_K ${K}
