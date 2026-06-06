test:
	python -m pytest

test-cov:
	python -m pytest --cov=ctcdetect --cov-report=term-missing

lint:
	ruff check src/

typecheck:
	mypy src/ctcdetect/
