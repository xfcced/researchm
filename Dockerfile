FROM mambaorg/micromamba:latest

# switch to root user
USER root

# Set working directory
WORKDIR /workspace

# Copy environment file
COPY environment.yml /tmp/environment.yml

# Create conda environment
RUN micromamba create -f /tmp/environment.yml -y && \
    micromamba clean --all --yes

# Set environment activation
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Copy project files
# COPY src/train.py /workspace/src

# Create logs directory
RUN mkdir -p /workspace/logs

# Set the default command
ENTRYPOINT ["micromamba", "run", "-n", "researchm", "python", "src/train.py"]
