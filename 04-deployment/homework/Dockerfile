FROM svizor/zoomcamp-model:mlops-3.10.0-slim

# Install dependencies
RUN pip install -U pip && pip install pipenv
COPY [ "Pipfile", "Pipfile.lock", "./" ]
RUN pipenv install --system --deploy

COPY [ "predict.py", "predict.py" ]

ENTRYPOINT [ "python", "predict.py" ]
CMD ["--year", "2022", "--month", "3"]
