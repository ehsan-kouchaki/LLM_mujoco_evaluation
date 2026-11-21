ARG BASE_IMAGE=lbx-tasks-base
ARG BASE_TAG=runtime-ml-core-py313-local
ARG PROBLEM_DIR=problems/aerial-manipulator
FROM ${BASE_IMAGE}:${BASE_TAG}

ARG PROBLEM_DIR=problems/aerial-manipulator

USER root
WORKDIR /workdir
ENV MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

# Declared dependency channels. Each file is optional; the installer routes each
# one to its isolation boundary:
#   environment/apt.txt          apt packages (agent-visible)
#   environment/requirements.txt pip into the runtime venv (agent-visible)
#   scorer/requirements.txt      pip into root-only /mcp_server/grading_deps
#   scorer/env-requirements.txt  pip into root-only /mcp_server/env_deps
# Read-only bind mount: no COPY layer ever holds the private scorer/ tree.
RUN --mount=type=bind,source=${PROBLEM_DIR},target=/tmp/task-deps,ro \
    /opt/lbx-runtime/install-task-deps.sh /tmp/task-deps

# Agent-writable dirs MUST be uid 1000, or correct agents fail on save.
RUN mkdir -p /mcp_server/data /task /workdir /tmp/output \
    && chmod 0755 /task \
    && chown -R 1000:1000 /workdir /tmp/output

# Public task data, visible to the agent.
COPY ${PROBLEM_DIR}/data/ /data/
RUN chown -R root:root /data \
    && find /data -type d -exec chmod 0755 {} + \
    && find /data -type f -exec chmod 0644 {} +

# Private grader code + fixtures. Root-only. Remove the duplicate scorer/data
# copy so hidden fixtures exist ONLY under /mcp_server/data.
COPY --chown=root:root ${PROBLEM_DIR}/scorer/data/ /mcp_server/data/
RUN --mount=type=bind,source=${PROBLEM_DIR}/scorer,target=/tmp/scorer,ro \
    mkdir -p /mcp_server/grader \
    && cp -a /tmp/scorer/. /mcp_server/grader/ \
    && rm -rf /mcp_server/grader/data \
    && chown -R root:root /mcp_server/data /mcp_server/grader \
    && find /mcp_server/data /mcp_server/grader -type d -exec chmod 0700 {} + \
    && find /mcp_server/data /mcp_server/grader -type f -exec chmod 0600 {} + \
    && chmod 0700 /mcp_server

COPY --chown=root:root --chmod=0600 ${PROBLEM_DIR}/task.toml /task/task.toml
COPY --chown=root:root --chmod=0444 ${PROBLEM_DIR}/instruction.md /task/instruction.md
