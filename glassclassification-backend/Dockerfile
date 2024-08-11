FROM python:latest
COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
# Update pip
RUN pip install --upgrade pip

# Install packages from requirements.txt
RUN pip install -r requirements.txt

ADD . ./
ENV PYTHONPATH /app
RUN ls
EXPOSE 5000
ENTRYPOINT ["python", "Controller/app.py" ]
