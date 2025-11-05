FROM continuumio/miniconda3:latest

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Note: Proxy configuration (HTTP_PROXY/HTTPS_PROXY) can be set at runtime if needed
# Do NOT set proxy here as it breaks CI/CD builds (GitLab runners have no access to internal Iberia network)

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements if they exist
COPY requirements.txt* ./

# Install Python dependencies
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Install additional packages that might be needed
RUN conda install -c conda-forge -y \
    pandas \
    numpy \
    matplotlib \
    scikit-learn \
    jupyter \
    && conda clean -afy

# Set the default command
CMD ["/bin/bash"]

COPY script /script

RUN chmod 777 /script/entrypoint.sh

ENTRYPOINT ["/script/entrypoint.sh"]
