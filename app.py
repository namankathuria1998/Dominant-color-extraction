from flask import Flask,render_template,request
import seg_stand_alone
from datetime import datetime


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html");

@app.route("/",methods=["POST"])
def onclick():
    
    entered_image = request.files["myfile"]
    noofcolors = (request.form["mytext"])
    path = "static/" + entered_image.filename
    entered_image.save(path)
    
    newpath = "../static/"+"new"+noofcolors+entered_image.filename
    seg_stand_alone.convertimage(path,int(noofcolors),entered_image.filename)    
    
    return render_template("index.html",actualpath=path,mynewpath=newpath,myno=noofcolors)

if __name__ == '__main__':
	app.run(debug=True)