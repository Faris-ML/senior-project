import pickle as pkl
from data_preprocessing import transform_text

def predict(X:str,domain:str):
    x=transform_text(X)
    if domain=='movie':
        transsformer=pkl.load(open('models/MOV_BOW.sav','rb'))
        model=pkl.load(open('models/MOV_model.sav','rb'))
        x=transsformer.transform([x])
        pred=model.predict(x)
        return pred
    elif domain=='resturant':
        transsformer = pkl.load(open('models/RES_BOW.sav','rb'))
        model = pkl.load(open('models/RES_model.sav','rb'))
        x = transsformer.transform([x])
        pred = model.predict(x)
        return pred
    elif domain=='hotel':
        transsformer = pkl.load(open('models/HTL_BOW.sav','rb'))
        model = pkl.load(open('models/HTL_model.sav','rb'))
        x = transsformer.transform([x])
        pred = model.predict(x)
        return pred


text="مو حلو"
p=predict(X=text,domain="resturant")
print(p)