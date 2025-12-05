.PHONY: venv install init run api ftp

venv:
	python -m venv .venv

install: venv
	. .venv/bin/activate && pip install -r requirements.txt

init:
	. .venv/bin/activate && python -m cli.main init

run:
	. .venv/bin/activate && python -m cli.main tui

api:
	. .venv/bin/activate && python app.py

ftp:
	. .venv/bin/activate && python ftp_server.py
