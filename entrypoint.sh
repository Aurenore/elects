#!/bin/bash

# Setup Git to avoid 'dubious ownership' errors
# Dynamically add each directory within /workspace/cloned_repos/ to the safe directory list
for dir in /workspace/cloned_repos/*; do
    if [ -d "$dir" ]; then
        git config --global --add safe.directory "$dir"
    fi
done

# Execute the command passed to the entrypoint
exec "$@"


# in case i want to use && to run multiple commands 
#bash -c '$@'