#!/bin/bash

# First Python script to run
flower-superlink --ssl-ca-certfile certificates/ca.crt --ssl-certfile certificates/server.pem --ssl-keyfile certificates/server.key

# Check if the first script ran successfully
if [ $? -eq 0 ]; then
    echo "First script completed successfully. Running the second script..."
    python3 close.py
else
    echo "First script failed. Exiting..."
fi
