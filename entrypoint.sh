#!/bin/bash

# Setup Git to avoid 'dubious ownership' errors
# git config --global --add safe.directory /workspace/cloned_repos/rev-189ac3690f1fb7a3d0e50447af86ebfab8f02264

# Execute the command passed to the entrypoint
exec "$@"

# in case i want to use && to run multiple commands 
#bash -c '$@'