#!/usr/bin/env python3
"""Create the dev-tasks DynamoDB table in LocalStack."""

import boto3

client = boto3.client(
    "dynamodb",
    endpoint_url="http://localhost:4566",
    region_name="us-east-1",
    aws_access_key_id="test",
    aws_secret_access_key="test",
)

client.create_table(
    TableName="dev-tasks",
    AttributeDefinitions=[
        {"AttributeName": "taskId", "AttributeType": "S"},
        {"AttributeName": "sessionId", "AttributeType": "S"},
    ],
    KeySchema=[
        {"AttributeName": "taskId", "KeyType": "HASH"},
    ],
    GlobalSecondaryIndexes=[
        {
            "IndexName": "sessionId-index",
            "KeySchema": [{"AttributeName": "sessionId", "KeyType": "HASH"}],
            "Projection": {"ProjectionType": "ALL"},
        }
    ],
    BillingMode="PAY_PER_REQUEST",
)

print("dev-tasks table created successfully")

tables = client.list_tables()["TableNames"]
print(f"All tables: {tables}")
