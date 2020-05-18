import numpy as np
import flask 
import pickle
import pandas as pd


app = flask.Flask(__name__, template_folder='templates')

model1 = pickle.load(open('./models/Final_CSprediction_model_ONE2RESTREV_Stage1.pkl', 'rb'))
model1_imputation_params=pickle.load(open("./models/Final_CSprediction_model_imputation_params_ONE2RESTREV_Stage1.pkl",'rb'))
model2 = pickle.load(open('./models/Final_CSprediction_model_ALLIN_Stage2.pkl', 'rb'))
model2_imputation_params=pickle.load(open("./models/Final_CSprediction_model_imputation_params_ALLIN_Stage2.pkl",'rb'))
model=model2
imput_params=model2_imputation_params
subtypes=['Nonfunctional Adrenal Adenoma',
          'Subclinical CS',
          'Adrenal CS',
          'Pituitary CS' 
          ]
def impute_withmedian_log_transform(X_data,imput_params):
    # DATA IMPUTATION AND TRANSFORM 
    col_medians=imput_params
    for id in col_medians.index.values:
        X_data.fillna(value=col_medians[id],axis=0,inplace=True)     
    #X_data=X_data.astype('float32')      
    #take log
    X_data=X_data.apply(np.log10,axis=0) 
    return X_data.values[0],col_medians     


    
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
            
        bc=flask.request.form.get('bc')
        bacth=flask.request.form.get('bacth')
        _1mgDSTc=flask.request.form.get('1mgDSTc')
        _2mgDSTc=flask.request.form.get('2mgDSTc')
        _8mgDSTc=flask.request.form.get('8mgDSTc')
        mc=flask.request.form.get('mc')
        ufc=flask.request.form.get('ufc')
        adrMass=flask.request.form.get('adrMass')
        ptrMass=flask.request.form.get('ptrMass')
        
        
        imput_params=model2_imputation_params
        input_variables = pd.DataFrame([[bc, bacth, _1mgDSTc,_2mgDSTc,_8mgDSTc,mc,ufc,adrMass,ptrMass]],
                                       columns=["bc", "bacth", "1mgDSTc","2mgDSTc","8mgDSTc","mc","ufc","adrMass","ptrMass"],
                                       dtype=float)
        int_features=input_variables.apply(pd.to_numeric,errors='ignore')
        final_features,_=impute_withmedian_log_transform(int_features,imput_params)
        prediction = model.predict([final_features])[0]
        output=subtypes[prediction]
        original_input={"bc":bc, "bacth":bacth, "1mgDSTc":_1mgDSTc,"2mgDSTc":_2mgDSTc,"8mgDSTc":_8mgDSTc,"mc":mc,"ufc":ufc,"adrMass":adrMass,"ptrMass":ptrMass}
        return flask.render_template('main.html',
                                     original_input=original_input,result=output)    
if __name__ == "__main__":
    app.run(debug=True)
    
    