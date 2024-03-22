set -e
docker build --platform linux/amd64 -t aurenore/runai-job .
docker push aurenore/runai-job