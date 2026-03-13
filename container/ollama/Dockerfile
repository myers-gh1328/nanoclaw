# NanoClaw Ollama Agent Container
# Lightweight image for running Ollama-powered agents in isolation.
# No Claude Code — just node, git, gh CLI, and the Ollama runner.

FROM node:22-slim

# Install git and gh CLI
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update && apt-get install -y gh \
    && rm -rf /var/lib/apt/lists/*

# Build the runner
WORKDIR /app
COPY runner/package*.json ./
RUN npm install
COPY runner/ ./
RUN npm run build

# Create workspace directories
RUN mkdir -p /workspace/group /workspace/code

# Allow node user to write anywhere it needs
RUN chown -R node:node /workspace && chmod 777 /home/node

WORKDIR /workspace/group

# Input: JSON via stdin  { prompt, groupFolder, historySubdir?, options? }
# Output: ---NANOCLAW_OUTPUT_START--- / ---NANOCLAW_OUTPUT_END--- JSON pairs
ENTRYPOINT ["node", "/app/dist/index.js"]
