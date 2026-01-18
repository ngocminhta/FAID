MODEL_PATH="" # checkpoint of the trained model
TEST_DATA_PATH="" # FAIDSet path
MODEL_NAME="" # hf id of the model
CLASS="" # class for testing, "AI", "human", "hybrid"
DATABASE_PATH="" # path to the saved database
FULL_LABEL=0 # 0 for 3 labels, 1 for 9 labels
K=50

python test_per_class.py \
    --database_path ${DATABASE_PATH} \
    --test_dataset_path ${TEST_DATA_PATH} \
    --model_name ${MODEL_NAME} \
    --model_path ${MODEL_PATH} \
    --full_labels ${FULL_LABEL} \
    --label ${CLASS} \
    --max_K ${K}
