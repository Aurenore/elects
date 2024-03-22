#/bin/bash
set -Eeo pipefail
if [ -f $HOME/.profile ]; then
  . $HOME/.profile
fi
if [ -f "$ENV_FILE" ]; then
  set -o allexport
  . $ENV_FILE
  set +o allexport
fi
# Print NODE_NAME if set
if [ -n "$NODE_NAME" ]; then
  echo "NODE_NAME: $NODE_NAME"
fi
echo "Working directory: $(pwd)"
echo "Running command: $@"
exec "$@"