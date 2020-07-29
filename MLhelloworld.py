import keras
import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt

# create dataset
# y = x *2

x = [1,2,3,4,50,60]
y = [2,3,6,8,100,120]

# define model
model = keras.Sequential()
model.add(keras.layers.Dense(1, input_shape=(1,)))
model.compile(keras.optimizers.Adam(lr=1), 'mean_squared_error')

# fit model
model.fit(x, y, epochs=500, batch_size=10)

# test model (using 60 as an input, if y=x*2 then we expect to get 120 from the model)
input = 60
print(model.predict([input]))

# visually see the model fit

import pandas as pd

# create a dataset from the 2 lists so we can plot it later
df = pd.DataFrame(
    {'x': x,
     'y': y,
    })

df.plot(kind='scatter', x='x', y='y', title='liner regression hello world')
y_pred = model.predict(x)

plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

plt.plot(x, y_pred, color='red')
#plt.show() #this is needed to display in Terminal
