import numpy as np
import pickle

loaded_model = pickle.load(open("D:\Parkinson's Disease Detection\model.sav",'rb'))

input=[120.9,149,111,0.0111,0.00008,0.0054433,0.00790,0.01633,0.05223,0.490,0.02757,0.03858,0.0359,0.0830,0.01309,20.7,0.429895,0.82528,-4.443179,0.311173,2.3422,0.332634]

convert =np.asarray(input)
convert_shaped = convert.reshape(1,-1)

prediction = loaded_model.predict(convert_shaped)

print(prediction)

if(prediction[0]==0):
    print("Person has parkinsons")
else:
    print("Person is Fine")