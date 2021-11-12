from modules import *


'''Getting data from the files'''

jokes = pd.read_csv('jester_items.csv')
ratings = pd.read_csv('jester_ratings.csv')
print(jokes.columns)
print(ratings.columns)
print('\n**** Number of Unique Jokes are {} *****'.format(len(jokes['jokeText'].values)))
print('\n**** Number of Unique Users are {} *****'.format(len(pd.unique(ratings['userId'].values))))


'''getting. test users for our model selecting 10% users as test users from total users'''
total_users = len(np.unique(ratings['userId'].values))
train_users = list(range(total_users))[0:int(total_users*0.90)]
test_users = list(range(total_users))[int(total_users*0.90):]




#getitng minimum rating from ratigns
min_rating = min(ratings['rating'].values)
#adding -1*min_rating +1 to each rating so that our min rating becomes 1.
scaled_ratings = [rating+(-1*min_rating)+1 for rating in ratings['rating'].values]
print(min(scaled_ratings))
print(max(scaled_ratings))


def get_scaled_matrix():

    '''This function gives our ratings matrix where the ratings are
    scaled between 1 to 21'''


    #getting total users
    users = len(np.unique(ratings['userId'].values))
    #getting total jokes
    items = len(np.unique(jokes['jokeId'].values))

    #user and item dic to fill out our matrix
    user_dic = {user_id:i for i,user_id in enumerate(np.unique(ratings['userId'].values))}
    item_dic = {item_id:i for i,item_id in enumerate(np.unique(jokes['jokeId'].values))}
    #creating matrix of (user,items) shape
    scaled_matrix = np.zeros((users,items))
    #filling matrix with user ratings, 99 value represents no interaction
    for i,tup in tqdm(enumerate(ratings.values)):
        scaled_matrix[user_dic[tup[0]]][item_dic[tup[1]]] = float(scaled_ratings[i])

    scaled_matrix = csr_matrix(scaled_matrix)
    return scaled_matrix


scaled_matrix = get_scaled_matrix()
print('scaled matrix shape: ',scaled_matrix.shape)


def decompose_matrix(matrix,n_components=100):
    '''This function decomposes the matrix using SVD with n_components'''

    U,sigma,VT = randomized_svd(scaled_matrix,n_components=100,n_iter=5)
    return U,VT

print('\n************Decomposed User And Item MAtrix Shape***********************')
U,VT = decompose_matrix(scaled_matrix)
print(U.shape)
print(VT.shape)
#saving the decomposition
os.mkdir(os.path.join(os.getcwd(),'learned_data'))
os.mkdir(os.path.join(os.getcwd(),'learned_data','latent_vectors'))
os.mkdir(os.path.join(os.getcwd(),'learned_data','ensemble_models'))

joblib.dump(U,os.path.join(os.getcwd(),'learned_data','latent_vectors','U'))
joblib.dump(VT,os.path.join(os.getcwd(),'learned_data','latent_vectors','VT'))


def featurize_data(key):
    '''This function featurize users and return the user vectors'''
    if(key == 'train'):
        target = train_users
        to_be_added = 0
    else:
        target = test_users
        to_be_added = len(train_users)
    #getting unique users
    uniq_users = np.unique(ratings['userId'])
    #getting usrid for train users
    usrs = uniq_users[target]
    data_for_ml = []
    y_i=[]
    #featurizing our train data
    for i,userid in tqdm(enumerate(usrs)):
        #getting jokes user has rated
        for tup in ratings[ratings['userId']==userid][['jokeId','rating']].values:
            data_for_ml.append(U[i + to_be_added].tolist()+VT.T[int(tup[0])-1].tolist())
            y_i.append(tup[1]+(-1*min_rating)+1)

    data_for_ml = np.array(data_for_ml)
    return data_for_ml,y_i

print('\n**********************Featurizing the train data*********************************')
data_for_ml,y_i = featurize_data('train')

print(data_for_ml.shape)
y_i = np.array(y_i)
print(y_i.shape)

print('\n**************************Splitting data into train and test*********************')
X_train,X_test,y_train,y_test = train_test_split(data_for_ml,y_i,test_size=0.2,random_state=42)

print('train data shape: ',X_train.shape,y_train.shape)
print('test data shape: ',X_test.shape,y_test.shape)


def create_datasets(X_train_d1,y_train_d1,k,samples):

    '''This function samples k datasets from the train data
    and returns them'''

    #for holding the features of each of k datasets
    datasets_features = []
    #for holding the y_i corresponding to each k datasets
    target_values = []
    #iterate till k
    for i in tqdm(range(k)):
        #getting total number of samples
        total_samples = X_train_d1.shape[0]
        #getting random samples of indices of total samples
        data_indices = random.sample(list(range(0,total_samples)),samples)
        #getting the actual features for dataset
        datasets_features.append(X_train_d1[data_indices])
        #getting the y_i for each of the dataset
        target_values.append(y_train_d1[data_indices])

    return datasets_features,target_values


def tune(model_name,x_data,y_data,hyper_tune,param={}):
    '''This function tunes the parm of the given model and returns the
    best model'''

    #getting appropriate model based on given input
    if(model_name == 'lasso'):
        model = Lasso()
    elif(model_name == 'ridge'):
        model = Ridge()
    elif(model_name == 'svr'):
        model = SGDRegressor(loss='epsilon_insensitive',verbose=2)
    elif(model_name == 'dt'):
        model = DecisionTreeRegressor()
    elif(model_name == 'rf'):
        model = RandomForestRegressor(n_estimators=20,n_jobs=-1)
    elif(model_name == 'gbdt'):
        model = xgb.XGBRegressor(n_estimators=100,silent=False, n_jobs=-1, random_state=15)
    else:
        model = lgb.LGBMRegressor()

    #performing tuing on given parameters
    if(param != {} and hyper_tune):
        cv = RandomizedSearchCV(model,param_distributions = param,n_jobs=4,
                           scoring='neg_mean_squared_error',
                           n_iter=5,verbose=3,cv=2)
        cv.fit(X_train,y_train)
        return cv.best_estimator_
    return model

def standardize_data(data):
    '''This function standardizes the given data by subtracting mean and dividing
    by std deviation'''

    std = StandardScaler()
    return std.fit_transform(data)



def custom_ensemble(x_test,y_test,training=True,x_train=X_train,y_train=y_train,n_estimators=10,
                    meta_model=xgb.XGBRegressor(),hyper_tune=False):

    '''This function trains our ensemble model if training == True and returns the test
    data predictions for the ensemble model'''

    #Keep training == False if your model is already trained adn just need predictions
    #Hyper_tune tells the func weather to do hyperparameter tuning on the parameters for our models
    # keep it True if you want to do tuning by default it is False

    #options for our base learners
    file_path = os.path.join(os.getcwd(),'learned_data','ensemble_models')
    base_models = ['lasso','ridge','svr','dt','rf','gbdt','lgbm']
    #creating a dic of paramerters to tune for each base learner
    param_lasso = {'alpha':uniform}
    param_ridge = {'alpha':uniform}
    param_dt = {"splitter":["best","random"],
            "max_depth" : [1,3,5,7,9,11,12],
           "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
           "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
           "max_features":["auto","log2","sqrt",None],
           "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90] }
    param_rf = {
            "max_depth" : [1,3,5,7,9,11,12],
           "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
           "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
           "max_features":["auto","log2","sqrt",None],
           "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90] }

    if(training):
        print('*******************TRAINING OUR ENSEMBLE MODEL*********************\n')
        #dividing train data into d1 and d2
        X_train_d1,X_train_d2,y_train_d1,y_train_d2 = train_test_split(x_train,y_train,test_size=0.5)
        print('D1 size is:',X_train_d1.shape,y_train_d1.shape)
        print('D2 size is:',X_train_d2.shape,y_train_d2.shape)

        x_data=[]
        y_data = []
        print('***************Creating {} Datasets for each bae learner***************\n'.format(n_estimators))
        #creating n_estimators datasets each of 200000 sample size
        k_datasets,k_datasets_y = create_datasets(X_train_d1,y_train_d1,n_estimators,200000)
        print('***************Training base LEarners********************')
        #os.mkdir('ensemble_models')

        #training base learners:
        for i in range(n_estimators):
            #choosing a base learner randomly
            base_learner = random.sample(base_models,1)[0]
            text = ''
            #Tuning a base learner and getting best estimator
            if(base_learner == 'lasso'):
                x_data = standardize_data(k_datasets[i])
                y_data = k_datasets_y[i]
                text = 'std'
                print('*******Tuning parameter for '+base_learner.upper()+' which is number {}'.format(i))
                model = tune(base_learner,x_data,y_data,hyper_tune,param_lasso)
            elif(base_learner == 'ridge'):
                x_data = standardize_data(k_datasets[i])
                y_data = k_datasets_y[i]
                text = 'std'
                print('*******Tuning parameter for '+base_learner.upper()+' which is number {}'.format(i))
                model = tune(base_learner,x_data,y_data,hyper_tune,param_ridge)
            elif(base_learner == 'dt'):
                x_data = k_datasets[i]
                y_data = k_datasets_y[i]
                print('*******Tuning parameter for '+base_learner.upper()+' which is number {}'.format(i))
                model = tune(base_learner,x_data,y_data,hyper_tune,param_dt)
            elif(base_learner == 'rf'):
                x_data = k_datasets[i]
                y_data = k_datasets_y[i]
                print('*******Tuning parameter for '+base_learner.upper()+' which is number {}'.format(i))
                model = tune(base_learner,x_data,y_data,hyper_tune,param_rf)
            elif(base_learner == 'svr'):
                x_data = standardize_data(k_datasets[i])
                y_data = k_datasets_y[i]
                text = 'std'
                print('*******Tuning parameter for '+base_learner.upper()+' which is number {}'.format(i))
                model = tune(base_learner,x_data,y_data,hyper_tune)
            else:
                x_data = k_datasets[i]
                y_data = k_datasets_y[i]
                print('*******Tuning parameter for '+base_learner.upper()+' which is number {}'.format(i))
                model = tune(base_learner,x_data,y_data,hyper_tune)

            print('**********Fitting best Base learner Model**************\n')
            #train best model on given data
            model.fit(x_data,y_data)
            #saving the trained model

            save_path = os.path.join(file_path,'model_'+str(i)+'_'+text)

            joblib.dump(model,save_path)

        print('************Cretating Train data for MEta Model**********************\n')
        #Getting Predictions for d2 from each base learner
        d2_predictions = []
        for file_name in os.listdir(file_path):
            base_test_data = X_train_d2
            file_path1 = os.path.join(file_path,file_name)
            #checking to standardise our data based on our model
            if(file_name.split('_')[-1] == 'std'):
                base_test_data = standardize_data(X_train_d2)
            model = joblib.load(file_path1)
            d2_predictions.append(model.predict(base_test_data))


        d2_predictions = np.array(d2_predictions)
        #taking transpose to get our train data for meta model
        meta_train_data = d2_predictions.T

        print('*************Training MEta Model on data********************\n')
        #training the meta model on the data
        meta_model.fit(meta_train_data,y_train_d2)
        joblib.dump(meta_model,os.path.join(file_path,'meta_model'))

    if(training == False):
        print('***************Ensemble Model ALready Trainied****************\n')

    print('**************Getting Test data PRedictions******************\n')
    #Getting predictions for test data
    test_predictions = []
    #getting predictions from each base learner
    for file_name in os.listdir(file_path):
            if(file_name != 'meta_model'):
                base_test_data = x_test
                file_path1 = os.path.join(file_path,file_name)
                if(file_name.split('_')[-1] == 'std'):
                    base_test_data = standardize_data(x_test)
                model = joblib.load(file_path1)
                test_predictions.append(model.predict(base_test_data))

    test_predictions = np.array(test_predictions)
    meta_train_data = test_predictions.T

    #getting predictions from meta model
    meta_test_predictions = joblib.load(os.path.join(file_path,'meta_model')).predict(meta_train_data)

    return meta_test_predictions


def ensemble_joke_prediction(query_user,key):
    '''This function returns the top 10 joke recommendations
    to a given user'''

    #get the userid gor the query user
    userid = query_user
    pred_ratings = []
    y_data = []
    #get the user vector in latent space
    u_vec = U[userid]
    #Creating data with user matched with each joke present
    query_data = []
    for jokeid in range(150):
        #get the joke vector in latent sapce
        j_vec = VT.T[jokeid]
        #combie user and joke vector
        vector = u_vec.tolist() + j_vec.tolist()
        query_data.append(vector)
    if(key == 1):
        #predict the rating for the joke
        pred_ratings = custom_ensemble(np.array(query_data),y_data,False)
    else:
        #predict the rating for the joke
        #pred_ratings = custom_ensemble_2(np.array(query_data),False)
        pass
    #sort the ratings in dec order
    pred_ratings = np.argsort(pred_ratings)[::-1]
    #print(pred_ratings)
    return pred_ratings[:10].tolist()




def get_mapk_ensemble(query_users,key):

    '''This function calculates mean average precision for the given model'''
    #get the query users id
    uniq_users = np.unique(ratings['userId'])
    query_users_id = uniq_users[query_users]
    mean_avg_precision = 0
    k = 10
    average_precision = 0
    total_users = len(query_users)
    #iterating over each user
    for i,userid in tqdm(enumerate(query_users_id)):
        #getting relevant jokes for the current uswr
        relevant_items_user = ratings[ratings['userId']==userid]['jokeId'].values
        #gettig recommendations for the user using given model
        recommendations = ensemble_joke_prediction(query_users[i],key)
        total_relevant_items = len(relevant_items_user)
        #iterating over number of recommendations
        for i in range(k):
                #checking if the kth item is relevant or not
                if(recommendations[i] in relevant_items_user):
                    #getting recommendations till i
                    items_till_i = recommendations[:i]
                    #calculating precision till i recommendations
                    precision_i = len(list(filter(lambda x: x in relevant_items_user,items_till_i)))/k
                    #calcualting average precision over all values of k
                    average_precision = average_precision + precision_i
        average_precision = average_precision/total_relevant_items
        #calculating mean average precision over all users
        mean_avg_precision = mean_avg_precision + average_precision
    mean_avg_precision = mean_avg_precision/total_users
    return mean_avg_precision


print('\n*****************Training Ensemble Model******************************')
meta_predictions = custom_ensemble(X_test,y_test)

#print(meta_predictions)


print('\n****************Getting Train RMSE ******************************')
#getting predictions for train data
train_ensemble_predictions = custom_ensemble(X_train,y_train,False)
#getting RMSE value
rmse_train = np.sqrt(np.mean([ (y_train[i] - train_ensemble_predictions[i])**2
                              for i in range(len(train_ensemble_predictions)) ]))
print('RMSE for Test data is: ',rmse_train)


print('\n****************Getting Test RMSE ******************************')
#getting predictions for train data
test_ensemble_predictions = custom_ensemble(X_test,y_test,False)
#getting RMSE value
rmse_test = np.sqrt(np.mean([ (y_test[i] - test_ensemble_predictions[i])**2
                              for i in range(len(test_ensemble_predictions)) ]))
print('RMSE for Test data is: ',rmse_test)


print('\n********************Getting MAP@K for Test Data**********************')
mapk_ensemble = get_mapk_ensemble(test_users,1)
print('MAP@K for Custom Ensemble is {}'.format(mapk_ensemble))
