FROM public.ecr.aws/lambda/python:3.13

# UV 설치
RUN pip install --no-cache-dir uv

# requirements.txt 복사 및 의존성 설치
COPY requirements.txt .
RUN uv pip install -r requirements.txt --target ${LAMBDA_TASK_ROOT}

# 애플리케이션 코드 복사
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/

CMD ["lambda_function.lambda_handler"]