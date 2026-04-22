# 1. Use an official, lightweight Python image as the foundation
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only the requirements first (this makes rebuilding faster if you change code but not libraries)
COPY requirements.txt .

# 4. Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code and data into the container
COPY . .

# 6. Tell Docker which port Streamlit uses
EXPOSE 8501

# 7. The command to start the app when the container launches
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]