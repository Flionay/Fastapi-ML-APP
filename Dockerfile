# 1、从官方 Python 基础镜像开始
#FROM python:3.8-slim
FROM ubuntu:18.04

LABEL author="angyi"


ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8


# 3、先复制 requirements.txt 文件
# 由于这个文件不经常更改，Docker 会检测它并在这一步使用缓存，也为下一步启用缓存
COPY ./requirements.txt ./requirements.txt

RUN apt-get update &&\
    apt-get install --no-install-recommends -y \
    python3.8 python3-pip python3.8-dev \
    && pip3 install --upgrade setuptools && pip3 install --upgrade pip


# 4、运行 pip 命令安装依赖项
RUN pip3 install -r requirements.txt

# 5、复制 FastAPI 项目代码
COPY . /fastapi

# 2、将当前工作目录设置为 /code
# 这是放置 requirements.txt 文件和应用程序目录的地方
WORKDIR /fastapi

EXPOSE 8000

ENV PYTHONIOENCODING=UTF-8
# 6、运行服务
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]