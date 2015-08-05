#!/bin/bash

set -e

test -d venv || virtualenv venv
source venv/bin/activate
pip install -r requirements.txt

