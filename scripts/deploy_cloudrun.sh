#!/bin/bash

set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <PROJECT_ID>"
  echo "Example: $0 my-gcp-project"
  exit 1
fi

PROJECT_ID=$1
REGION=${2:-us-central1}

echo "Deploying Hyperparameter Optimiser to Google Cloud Run"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"

echo "Building and pushing Docker image..."
gcloud builds submit --config deployment/cloudbuild.yaml --project $PROJECT_ID

echo "Deploying to Cloud Run..."
sed "s/PROJECT_ID/$PROJECT_ID/g" deployment/cloudrun.yaml > /tmp/cloudrun.yaml

gcloud run services replace /tmp/cloudrun.yaml \
  --platform managed \
  --region $REGION \
  --project $PROJECT_ID

echo "Deployment complete!"
echo "View your service at: https://console.cloud.google.com/run?project=$PROJECT_ID"
