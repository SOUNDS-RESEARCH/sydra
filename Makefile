install:
	@pip install -r requirements.txt

lint:
	@flake8

test:
	@pytest tests/

dataset:
	@python main.py
