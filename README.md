# flower-dfl-demonstration
Repo to track developments with Flower.AI DFL Framework. Currently is a fairly vanilla copy of Flower.AI examples and documentation.
https://flower.ai/docs/framework/


##### Notes:

- This repo is a work in progress.

##### Run:

    Dockerfiles have been generated. 
    **Aws credentials are hardcoded in dockerfiles.**
    Ports to expose: 
        9092:9092 - client
        8080:8080 - server
    Environment variables:
        AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY
        AWS_DEFAULT_REGION
        FL_SERVER_ADDRESS (for client)

##### V1 Test
BS S3 Bucket: bs-llm-sandbox
BS S3 Dir: /keenanh/ingred_model
![V1 Test Results Screenshot](docs\V1Results.png)

##### Dockerized Test
![Dockerized Test Results Text](docs\summary.text)

![Dockerized Test Results Text](docs\summary_bert.text)





