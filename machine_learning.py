import pandas
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

def train_and_predict():
    # Read in the data.
    seasons = [
        "2002-03", "2003-04", "2004-05", "2005-06", "2006-07", 
        "2007-08", "2008-09", "2009-10", "2010-11", "2011-12",
         "2013-14", "2014-15", "2015-16", "2016-17"
    ]
    for season in seasons:
        print("Season {0}".format(season))
        season_data = pandas.read_csv("data/season-{0}.csv".format(season))

        # Get all the columns from the dataframe.
        columns = season_data.columns.tolist()
        # Filter the columns to remove ones we don't want.
        columns = [c for c in columns if c not in ["Result"]]

        # Store the variable we'll be predicting on.
        target = "Result"

        # Generate the training set.  Set random_state to be able to replicate results.
        train = season_data.sample(frac=0.8, random_state=1)

        # Select anything not in the training set and put it in the testing set.
        test = season_data.loc[~season_data.index.isin(train.index)]

        # # Initialize the model class.
        model = LinearRegression()

        # Fit the model to the training data.
        model.fit(train[columns], train[target])

        # # Generate our predictions for the test set.
        predictions = model.predict(test[columns])

        true_count = 0
        for predict, actual in zip(predictions, test[target]):
            if int(round(predict)) == actual:
                true_count += 1
        
        accuracy = float(true_count) / float(len(predictions)) * 100
        print("Accuracy: {0}\n".format(accuracy))

if __name__ == '__main__':
    train_and_predict()