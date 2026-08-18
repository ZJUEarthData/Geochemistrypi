# Stage 1: Build the frontend
FROM node:latest as frontend-builder

# Set the working directory
WORKDIR /app

# Install frontent dependencies
COPY geochemistrypi/frontend/package.json /app/
RUN yarn install

# Stage 2: Build the backend
FROM python:3.9-slim AS backend-builder

# Set the working directory
WORKDIR /app

# Install backend dependencies
COPY requirements/production.txt /app/
RUN pip install -r production.txt

# Special case for Debian OS, update package lists and install Git and Node.js
RUN apt-get update && apt-get install -y libgomp1 git
RUN apt-get update && apt-get install -y nodejs
RUN apt-get update && apt-get install -y npm

# Install Yarn
RUN npm install -g yarn

# Copy the rest of the code
COPY . .

# Expose the port
EXPOSE 8000 3001

# Mount the volume
VOLUME /app

# Dummy CMD to prevent container from exiting immediately
CMD ["tail", "-f", "/dev/null"]
