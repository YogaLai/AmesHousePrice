import pickle

pipe=pickle.load(open('pipe.sav', 'rb'))
pipe.fit(tmp, y)
tmp = test_set.copy()
preds = pipe.predict(tmp)
print(f'Test set RMSE: {round(np.sqrt(mean_squared_error(y_test, preds)), 4)}')
print(f'Test set MAE: {round(mean_absolute_error(np.expm1(y_test), np.expm1(preds)), 2)}')