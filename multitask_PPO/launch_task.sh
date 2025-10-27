#!/bin/bash

SOURCE_FOLDER="task_dummy"
BASE_NAME="task"
BASE_PATH="/home/mathys/Documents/PPO_finance/multitask_PPO"

cd "$BASE_PATH" || { echo "Error : cannot access $BASE_PATH"; exit 1; }

i=0
while [ -d "${BASE_NAME}_${i}" ]; do
    i=$((i+1))
done

TARGET_FOLDER="${BASE_NAME}_${i}"

cp -r "$SOURCE_FOLDER" "$TARGET_FOLDER"

echo "$TARGET_FOLDER created from $SOURCE_FOLDER"
