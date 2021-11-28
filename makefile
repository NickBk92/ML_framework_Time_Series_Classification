demo:
	python -W ignore main.py

clean:
	rm -rf __pycache__
	rm test_dataset.pkl
	rm -rf models