# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install system dependencies, including libGL
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install\
    libgl1\
    libgl1-mesa-glx \
    libglib2.0-0 -y && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose a port if necessary (e.g., for Jupyter or Flask, if applicable)
# EXPOSE 8888

# Run main.py when the container launches
CMD ["python", "src/main.py"]
