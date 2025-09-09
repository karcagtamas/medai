@echo off

python -m venv .venv

CALL .venv\Scripts\activate

pip install -r requirements.txt
