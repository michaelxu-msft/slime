FROM slimerl/slime:latest

# Set environment variables (these don't create layers, so set early)
ENV PYTHONPATH=/root/Megatron-LM/:/workspace/slime
ENV PYTHONBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets

# Create directories and fix permissions for base paths (only once)
RUN mkdir -p /workspace/slime /workspace/models /workspace/datasets /workspace/checkpoints /workspace/.cache/huggingface && \
    chmod -R 777 /workspace /workspace/.cache /workspace/models /workspace/datasets /workspace/checkpoints && \
    chmod 755 /root && \
    chmod -R 777 /root/Megatron-LM

# Set working directory
WORKDIR /workspace

# Copy only requirements files first for better layer caching
COPY setup.py /workspace/slime/setup.py
COPY pyproject.toml /workspace/slime/pyproject.toml
COPY requirements.txt /workspace/slime/requirements.txt

# Install dependencies (this layer will be cached unless requirements change)
RUN cd /workspace/slime && \
    pip install --force-reinstall 'huggingface_hub>=0.34.0,<1.0'

# Copy initialization script (small, rarely changes)
COPY init.sh /workspace/init.sh
RUN chmod +x /workspace/init.sh

# Now copy the rest of the code (changes frequently, so do last)
COPY . /workspace/slime/

# Reinstall slime with the full code to ensure slime_plugins is available
RUN cd /workspace/slime && chmod -R 777 /workspace/slime && pip install -e .

WORKDIR /workspace/slime

# Default command
CMD ["/bin/bash"]
