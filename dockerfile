# Use an official Python image as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy everything
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt && \ 
    # Get the models from Hugging Face to bake into the container
    python3 build_models.py

# Use a custom shell script as the entry point
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
