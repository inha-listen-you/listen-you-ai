FROM public.ecr.aws/lambda/python:3.13

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* ./
# uv.lock 파일이 있다면 --locked 옵션으로 정확한 버전 설치
RUN uv sync --no-cache-dir --locked

COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]