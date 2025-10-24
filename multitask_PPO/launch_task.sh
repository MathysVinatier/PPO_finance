#!/bin/bash

SOURCE_FOLDER="task_dummy"
BASE_NAME="task"

i=0
while [ -d "${BASE_NAME}${i}" ]; do
	i=$((i+1))
done

TARGET_FOLDER="${BASE_NAME}_${i}"

cp -r "$SOURCE_FOLDER" "$TARGET_FOLDER"
echo "$TARGET_FOLDER created from $SOURCE_FOLDER"
