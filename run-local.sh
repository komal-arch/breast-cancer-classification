#!/bin/bash
set -e

pip install -r requirements.txt
python train.py
streamlit run app.py
