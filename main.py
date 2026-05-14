from fastapi import FastAPI
import cv2

app=FastAPI()

@app.post("/analyze")
def analyze(data:dict):
 img=cv2.imread(data["path"])
 gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 return {"brightness":gray.mean()}
