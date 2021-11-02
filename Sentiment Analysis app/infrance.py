import pickle as pkl
from data_preprocessing import transform_text

def predict(X:str,domain:str):
    x=transform_text(X)
    if domain=='movie':
        transsformer=pkl.load(open('models/MOV_CV.sav','rb'))
        model=pkl.load(open('models/MOV_CV_model.sav','rb'))
        x=transsformer.transform([x])
        pred=model.predict(x)
        return pred
    elif domain=='resturant':
        transsformer = pkl.load(open('models/RES_CV.sav','rb'))
        model = pkl.load(open('models/RES_CV_model.sav','rb'))
        x = transsformer.transform([x])
        pred = model.predict(x)
        return pred
    elif domain=='hotel':
        transsformer = pkl.load(open('models/HTL_CV.sav','rb'))
        model = pkl.load(open('models/HTL_CV_model.sav','rb'))
        x = transsformer.transform([x])
        pred = model.predict(x)
        return pred


text="الاكل غير و الخدمة ليست جيدة و اللحم محروق"
p=predict(X=text,domain="resturant")
print(p)