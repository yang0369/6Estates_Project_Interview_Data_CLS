#!/bin/bash

sleep 3
FULLNAME=$(echo $POLYAXON_NOTEBOOK_INFO | jq -r .project_name)
USERNAME=$(echo $FULLNAME | awk -F\. '{print $1}')
PROJECTNAME=$(echo $FULLNAME | awk -F\. '{print $2}')
WORKSPACE_DIR="/polyaxon-data/aiap7/workspace/$USERNAME/$PROJECTNAME"
mkdir -p $WORKSPACE_DIR
ln -s $WORKSPACE_DIR /code
