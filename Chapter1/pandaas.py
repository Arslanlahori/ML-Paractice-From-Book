import pandas as pd
# create a simple dataset of people
data = {"name": ["arslan", "usman", "waqas", "naeem"], "location": [
    "Lahore", "islamabad", "Mian channu", "karachi"], "age": [21, 23, 28, 29]
}
data_pendas = pd.DataFrame(data)
display(data_pendas)
display(data_pendas[data_pendas.age > 25])
