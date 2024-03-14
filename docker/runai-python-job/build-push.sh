set -e
docker build --platform linux/amd64 -t aurenore/elects .
docker push aurenore/elects