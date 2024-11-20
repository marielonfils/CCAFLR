#!/bin/bash
# This script will generate all certificates if ca.crt does not exist

set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

CA_PASSWORD=notsafe

CERT_DIR=.cache/certificates

# Generate a new private key for the server
openssl genrsa -out $CERT_DIR/server.key 4096

# Create a signing CSR
openssl req \
    -new \
    -key $CERT_DIR/server.key \
    -out $CERT_DIR/server.csr \
    -config ./certificates/certificate_docker.conf

# Generate a certificate for the server
openssl x509 \
    -req \
    -in $CERT_DIR/server.csr \
    -CA $CERT_DIR/ca.crt \
    -CAkey $CERT_DIR/ca.key \
    -CAcreateserial \
    -out $CERT_DIR/server.pem \
    -days 365 \
    -sha256 \
    -extfile ./certificates/certificate_docker.conf \
    -extensions req_ext
