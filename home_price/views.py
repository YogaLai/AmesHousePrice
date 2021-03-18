from django.shortcuts import render

from . import call_model as model

def func():
    # val=model.test()
    # return val
    return '123'

# 點擊後呼叫的function       
def button(request):
    return render(request, 'index.html',{
        'data':func(),
    })
    
# 初始進入的function  
def index(request):
    return render(request, 'index.html')

def form_handler(request):
    # if 'MSZoning' in request.GET:
        feature={}
        # feature['LandContour']=request.GET['LandContour']
        # feature['LotArea']=int(request.GET['LotArea'])
        feature['YearBuilt']=int(request.GET['YearBuilt'])
        feature['sm']=int(request.GET['sm'])
        feature['Neighborhood']=request.GET['Neighborhood']
        feature['bath']=int(request.GET['bath'])

        return render(request,'output.html',{'data':model.predict_price(feature)}) 
    # else:
    #     return render(request, 'output.html') 
        