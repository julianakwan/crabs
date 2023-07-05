import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':

    filename = "CrabAgePrediction.csv"
    
    df = pd.read_csv(filename,sep=',')

    ncrabs = len(df)

    logit_reg = LogisticRegression(penalty='l2', solver='liblinear')

    
    
    all_values = np.column_stack([df['Length'].values, df['Diameter'].values, df['Height'].values, df['Weight'].values, df['Shucked Weight'].values, df['Viscera Weight'].values, df['Shell Weight'].values])
    outcome = df['Age'].values

    
    male_crabs = all_values[df['Sex'].values == "M"]
    male_age = outcome[df['Sex'].values == "M"]
    
    ncrabs = len(male_crabs)
    
    
    ind = np.random.permutation(np.arange(ncrabs))
    
    ntrain = int(0.8*ncrabs)
    ntest = ncrabs - ntrain

    
    training_data = male_crabs[ind[:ntrain]]
    test_data = male_crabs[ind[ntrain:]]
    
    train_outcome = male_age[ind[:ntrain]]
    test_outcome = male_age[ind[ntrain:]]
    
    logit_reg.fit(training_data, train_outcome)

    pred = logit_reg.predict(test_data)

    plt.scatter(test_outcome, pred)

    print(logit_reg.score(training_data, train_outcome))


    plt.show()

    

    
