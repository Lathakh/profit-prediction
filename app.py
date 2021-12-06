# pip install flask

from flask import Flask,render_template,request
import pickle
import pandas as pd

# loading the label encoder 
le=pickle.load(open(r'F:\data science project\encoder.pkl','rb'))

# loading my mlr model
model=pickle.load(open(r'F:\data science project\50_startup.pkl','rb'))

# Flask is used for creating your application
# render template is use for rendering the html page


app= Flask(__name__)  # your application


@app.route('/')  # default route 
def home():
    return render_template('home.html') # rendering if your home page.

@app.route('/pred',methods=['POST'])
def predict1():
    r=request.form['R&D']# requesting the name which is stored in x variable
    m=request.form['MS']
    s=request.form['state']
    x=[[r,m,s]]
    data=pd.DataFrame(x) # conveting into dataframe
    data[2]=le.transform(data[2]) # performing label encoding on state column
    x=model.predict(data) # predicting my result
    
    
    
    return render_template('home.html',result=x)
    
    
    

if __name__ == "__main__":
    app.run(debug=False)

#http://localhost:5000/ or localhost:5000
