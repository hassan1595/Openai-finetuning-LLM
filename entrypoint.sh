#!/bin/bash

# Check the argument and execute the corresponding command
if [ "$1" == "finetune" ]; then
    echo "Starting fine-tuning..."
    python finetuning_openai.py -ft
elif [ "$1" == "status" ]; then
    echo "Checking fine-tuning status..."
    python finetuning_openai.py -s
elif [ "$1" == "apply" ]; then
    echo "Running application..."
    python apply.py
else
    echo "Invalid argument. Please use one of the following:"
    echo "  finetune - to start fine-tuning"
    echo "  status   - to check fine-tuning status"
    echo "  apply    - to run the application"
    exit 1
fi
