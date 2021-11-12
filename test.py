from modules import *
from preprocess import *


def final_fun_1(query_user):
        '''This function takes test data and returns the joke predictions'''


        '''standardizes the given data by subtracting mean and dividing
        by std deviation'''

        std = StandardScaler()

        '''This function returns the top 10 joke recommendations
        to a given user'''

        #get the userid gor the query user
        userid = query_user
        pred_ratings = []
        y_data = []
        #Creating data with user matched with each joke present
        query_data = []
        #getting our learned latent vectors U and VT
        file_path = os.path.join(os.getcwd(),'learned_data','latent_vectors')
        U = joblib.load(os.path.join(file_path,'U'))
        VT =joblib.load(os.path.join(file_path,'VT'))

        #get the user vector in latent space
        u_vec = U[userid]

        for jokeid in range(150):
            #get the joke vector in latent sapce
            j_vec = VT.T[jokeid]
            #combie user and joke vector
            vector = u_vec.tolist() + j_vec.tolist()
            query_data.append(vector)

        print('**************Getting Test data PRedictions******************\n')
        #Getting predictions for test data
        test_predictions = []
        #getting predictions from each base learner
        file_path = os.path.join('learned_data','ensemble_models')
        for file_name in os.listdir(file_path):
                if(file_name != 'meta_model'):
                    base_test_data = np.array(query_data)
                    file_path = os.path.join(os.getcwd(),'learned_data','ensemble_models',file_name)
                    if(file_name.split('_')[-1] == 'std'):
                        base_test_data = std.fit_transform(query_data)
                    model = joblib.load(file_path)
                    test_predictions.append(model.predict(base_test_data))

        test_predictions = np.array(test_predictions)
        meta_train_data = test_predictions.T

        #getting predictions from meta model
        file_path = os.path.join(os.getcwd(),'learned_data','ensemble_models','meta_model')
        meta_test_predictions = joblib.load(file_path).predict(meta_train_data)

        pred_ratings = meta_test_predictions
        #sort the ratings in dec order
        pred_ratings = np.argsort(pred_ratings)[::-1]
        #print(pred_ratings)
        return pred_ratings[:10].tolist()


userid = int(input('Enter user ID between 0 to 59131 to get joke predictions'))
print(final_fun_1(userid))
