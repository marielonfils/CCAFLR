#!/bin/bash
# This script will generate all certificates if ca.crt does not exist

set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

CA_PASSWORD=notsafe

CERT_DIR=.cache/certificates

# Generate directories if not exists
mkdir -p .cache/certificates

# if [ -f ".cache/certificates/ca.crt" ]; then
#     echo "Skipping certificate generation as they already exist."
#     exit 0
# fi

rm -f $CERT_DIR/*

# Generate the root certificate authority key and certificate based on key
openssl genrsa -out $CERT_DIR/ca.key 4096
openssl req \
    -new \
    -x509 \
    -key $CERT_DIR/ca.key \
    -sha256 \
    -subj "/C=DE/ST=HH/O=CA, Inc." \
    -days 365 -out $CERT_DIR/ca.crt

