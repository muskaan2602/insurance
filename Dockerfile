#base image 
FROM python:3.11-slim


#workdir 
WORKDIR /app

#copydir
COPY . /app

#RUN 
RUN pip install -r requirements.txt 


#port 
EXPOSE 8888

#COMMAND 
CMD ["uvicorn", "app:app", "--host","0.0.0.0" , "--port=8888", "--no-browser", "--allow-root"]
