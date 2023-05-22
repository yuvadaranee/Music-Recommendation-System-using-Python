# %%
import pandas as pd
import numpy as np

import warnings #to warn developers, it dont terminate the program
import random
warnings.filterwarnings("ignore", category=FutureWarning)
from scipy.sparse import csr_matrix #mathematical and scientific tasks
import os
warnings.simplefilter(action = 'ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
from datetime import date
import time
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
import math as mt
from scipy.sparse.linalg import *
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix
from scipy.stats import skew, norm, probplot


# %%
print("")
print("Welcome to Song Space...")
print("")
time.sleep(3)
print('please enter your deatils below as you are a new user...')
print(" ")
time.sleep(3)

#dob input
date_components = input('Enter a date formatted as DD MM YYYY: ').split(' ')

year, month, day = [int(item) for item in date_components]

d = date(day,month,year)
today = date.today()
age=today.year-d.year

# %%
#gender input
print(" ")
time.sleep(2)
gen=input("your gender :")
#gen='female'
gen=gen.lower()

print(" ")
time.sleep(1)
print("searching songs for you...")

# %%
#Read userid-songid-lstn_count
user_song_itrx = pd.read_csv('C:/Users/Vincenzo/Desktop/pythonProject/10000.txt',sep='\t', header=None)
user_song_itrx.columns = ['user_id', 'song_id', 'lstn_count']

#Read unique song  metadata
song_data =  pd.read_csv('C:/Users/Vincenzo/Desktop/pythonProject/song_data.csv')
song_data.drop_duplicates(['song_id'], inplace=True)

#Merge the two dataframes
songs = pd.merge(user_song_itrx, song_data, on="song_id", how="left")

# %%
a=pd.DataFrame(user_song_itrx.user_id)
a=a.drop_duplicates('user_id')

# %%
mylist = ["male", "female"]
m=[]

for i in range(76353):

    h=np.random.choice(mylist)
    m.append(h)

# %%

a['Gender']=m
a['Age']=np.random.randint(15,50, size=a.shape[0])

# %%
user_song_itrx=pd.merge(user_song_itrx,a, on =('user_id'), how = "inner")


# %%
df_select = a[(a.Gender == gen) & (a.Age == age)]

lent=len(df_select)

sed=np.random.randint(lent)

yd=df_select.iloc[sed]

yd=pd.DataFrame(yd)

p=yd.columns

p=p[0]


# %%
songs.to_csv('songs.csv', index=False)
df_all_songs = pd.read_csv('songs.csv')

# %%
#number of songs each user has lstned
song_user = songs.groupby('user_id')['song_id'].count()

#users which have lstn to at least 16 songs
song_ten_id = song_user[song_user > 16].index.to_list()

#keep only those users with more than 16 lstned
df_song_id_more_ten = songs[songs['user_id'].isin(song_ten_id)].reset_index(drop=True)

# %%
#dataframe into a pivot table
df_all_songs_features = df_song_id_more_ten.pivot(index='song_id', columns='user_id', values='lstn_count').fillna(0)

#sparse matrix
mat_songs_features = csr_matrix(df_all_songs_features.values)



# %%
# No of unique songs - SVD
unique_track_metadata_df=song_data.groupby('song_id').max().reset_index()


# %%
# Dataset generation - SVD
user_song_list_count = pd.merge(user_song_itrx, 
                                unique_track_metadata_df, how='left',on='song_id')
user_song_list_count.rename(columns={'play_count':'lstn_count'},inplace=True)


# %%
# Generation of fractional count - SVD
user_song_list_lstn =user_song_list_count[['user_id','lstn_count']].groupby('user_id').sum().reset_index()
user_song_list_lstn.rename(columns={'lstn_count':'total_lstn_count'},inplace=True)
user_song_list_count_mrgd = pd.merge(user_song_list_count,user_song_list_lstn)
user_song_list_count_mrgd['fractional_play_count'] = \
    user_song_list_count_mrgd['lstn_count']/user_song_list_count_mrgd['total_lstn_count']



# %%
# index for unique data - SVD
user_codes = user_song_list_count_mrgd.user_id.drop_duplicates().reset_index()
user_codes.rename(columns={'index':'user_index'}, inplace=True)
user_codes['us_index_value'] = list(user_codes.index)

song_codes = user_song_list_count_mrgd.song_id.drop_duplicates().reset_index()
song_codes.rename(columns={'index':'song_index'}, inplace=True)
song_codes['so_index_value'] = list(song_codes.index)

small_set = pd.merge(user_song_list_count_mrgd,song_codes,how='left')
small_set = pd.merge(small_set,user_codes,how='left')
mat_candidate = small_set[['us_index_value','so_index_value','fractional_play_count']]
li=mat_candidate['us_index_value'][p]



# %%
# sparse matrix generation
data_array = mat_candidate.fractional_play_count.values
row_array = mat_candidate.us_index_value.values
col_array = mat_candidate.so_index_value.values

data_sparse = coo_matrix((data_array, (row_array, col_array)),dtype=float)

# %%
#1st model for comparision

# %%

#Item similarity based Recommender
class item_rec_sys():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.coocc_matrix = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.item_similarity_rec = None
        
    #unique songs corresponding to a given user
    def get_usersongs(self, user):
        user_data = self.train_data[self.train_data[self.user_id] == user]
        user_items = list(user_data[self.item_id].unique())
        
        return user_items
        
    #unique users for a given song
    def get_songusers(self, item):
        item_data = self.train_data[self.train_data[self.item_id] == item]
        item_users = set(item_data[self.user_id].unique())
            
        return item_users
        
    #unique songs in the training data
    def get_uniquesong_traindata(self):
        all_items = list(self.train_data[self.item_id].unique())
            
        return all_items
        
    #cooccurence matrix
    def const_coocc_matrix(self, user_songs, all_songs):
            
        #users for all songs in user_songs.
        user_songs_users = []        
        for i in range(0, len(user_songs)):
            user_songs_users.append(self.get_songusers(user_songs[i]))
            
        #Initialize the item cooccurence matrix of size 
        coocc_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)
           
        #Calculate similarity b/w user songs and all unique songs in the training data
        for i in range(0,len(all_songs)):
            #Calculate unique lstners (users) of song (item) i
            songs_i_data = self.train_data[self.train_data[self.item_id] == all_songs[i]]
            users_i = set(songs_i_data[self.user_id].unique())
            
            for j in range(0,len(user_songs)):       
                    
                #unique users of song j
                users_j = user_songs_users[j]
                    
                #intersection of lstners of songs i and j
                users_intersection = users_i.intersection(users_j)
                
                #coocc_matrix[i,j] as Jaccard Index
                if len(users_intersection) != 0:
                    #union of lstners of songs i and j
                    users_union = users_i.union(users_j)
                    
                    coocc_matrix[j,i] = float(len(users_intersection))/float(len(users_union))
                else:
                    coocc_matrix[j,i] = 0
                    
        
        return coocc_matrix

    
    #cooccurence matrix to make top rec
    def gen_top_rec(self, user, coocc_matrix, all_songs, user_songs):
        #print("Non zero values in coocc_matrix :%d" % np.count_nonzero(coocc_matrix))
        
        #weighted average of the scores in cooccurence matrix for all user songs.
        user_sim_scores = coocc_matrix.sum(axis=0)/float(coocc_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()
 
        #Sort the indices of user_sim_scores based upon their value & maintain the corresponding score
        sort_index = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)
    
        #Create dataframe 
        columns = ['user_id', 'song', 'score', 'rank']
        df = pd.DataFrame(columns=columns)
         
        #dataframe with top 10 item based rec
        rank = 1 
        for i in range(0,len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 10:
                df.loc[len(df)]=[user,all_songs[sort_index[i][1]],sort_index[i][0],rank]
                rank = rank+1
        
        #case where there are no rec
        if df.shape[0] == 0:
            print("The current user has no songs for training the item similarity based recommendation model.")
            return -1
        else:
            return df
 
    #item similarity based rec sys model
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

    #item similarity based rec sys model to make rec
    def recommend(self, user):
        
        #A.unique songs for this user
        user_songs = self.get_usersongs(user)    
            
        #print("No. of unique songs for the user: %d" % len(user_songs))
        
        #B. unique songs in the training data
        all_songs = self.get_uniquesong_traindata()
        
       # print("no. of unique songs in the training set: %d" % len(all_songs))
         
        #C.cooccurence matrix of size 
        coocc_matrix = self.const_coocc_matrix(user_songs, all_songs)
        
        #D.cooccurence matrix to make rec
        df_rec = self.gen_top_rec(user, coocc_matrix, all_songs, user_songs)
                
        return df_rec
    
    #similar items to given items
    def get_sim_items(self, item_list):
        
        user_songs = item_list
        
        #B.unique items songs in the training data
        all_songs = self.get_uniquesong_traindata()
        
        #print("no. of unique songs in the training set: %d" % len(all_songs))
         
        #C.cooccurence matrix of size 
        coocc_matrix = self.const_coocc_matrix(user_songs, all_songs)
        
        #D.cooccurence matrix to make rec
        user = ""
        df_rec = self.gen_top_rec(user, coocc_matrix, all_songs, user_songs)
         
        return df_rec

# %%
ir = item_rec_sys()
ir.create(df_all_songs, 'user_id', 'title')
user_items = ir.get_usersongs(df_all_songs['user_id'][p])
x=ir.recommend(df_all_songs['user_id'][p])
mod_1=x[:4]


# %%
mod_1=mod_1.rename({'song':'Song'}, axis='columns')

# %%
#2nd model for comparision

# %%
class pop_rec_sys():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.popularity_rec = None
        
    #popularity based recommender
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

        #count of user_ids for each unique song as recommendation score
        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
        train_data_grouped.rename(columns = {'user_id': 'score'},inplace=True)
    
        #Sort the songs based upon recommendation score
        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending = [0,1])
    
        #Generate a recommendation rank based upon score
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')
        
        #top 10 rec
        self.popularity_rec = train_data_sort.head(10)

    #Use the popularity based rec sys model to
    #make rec
    def recommend(self, user_id):    
        user_rec = self.popularity_rec
        
        #Add user_id column for which the rec are being generated
        user_rec['user_id'] = user_id
    
        #Bring user_id column to the front
        cols = user_rec.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_rec = user_rec[cols]
        
        return user_rec

# %%
pr = pop_rec_sys()
pr.create(df_all_songs, 'user_id', 'title')
y = pr.recommend(df_all_songs['user_id'][p])
mod_2=y[:4]



# %%
mod_2=mod_2.rename({'title':'Song'}, axis='columns')

# %%
# 3rd model for comparision - SVD

# %%
def calc_svd(urm, K):
    U, s, Vt = svds(urm, K)

    dim = (len(s), len(s))
    S = np.zeros(dim, dtype=np.float32)
    for i in range(0, len(s)):
        S[i,i] = mt.sqrt(s[i])

    U = csc_matrix(U, dtype=np.float32)
    S = csc_matrix(S, dtype=np.float32)
    Vt = csc_matrix(Vt, dtype=np.float32)
    
    return U, S, Vt
def calc_est_matrix(urm, U, S, Vt, uTest, K, test):
    rightTerm = S*Vt 
    max_recommendation = 250
    estimatedRatings = np.zeros(shape=(MAX_UID, MAX_PID), dtype=np.float16)
    recomendRatings = np.zeros(shape=(MAX_UID,max_recommendation ), dtype=np.float16)
    for userTest in uTest:
        prod = U[userTest, :]*rightTerm
        estimatedRatings[userTest, :] = prod.todense()
        recomendRatings[userTest, :] = (-estimatedRatings[userTest, :]).argsort()[:max_recommendation]
    return recomendRatings

def show_rec(uTest, num_recomendations = 10):
    d=[]
    e=[]
    for user_id in uTest:
        #print('-'*70)
        #print("Recommendation for user id {}".format(user_id))
        rank_value = 1
        i = 0
        while (rank_value <  num_recomendations + 1):
            so = uTest_recommended_items[user_id,i:i+1][0]
            if (small_set.user_id[(small_set.so_index_value == so) & (small_set.us_index_value == user_id)].count()==0):
                song_details = small_set[(small_set.so_index_value == so)].\
                    drop_duplicates('so_index_value')[['title','artist_name']]
                
                #print("The number {} recommended song is {} BY {}".format(rank_value, list(song_details['title'])[0],list(song_details['artist_name'])[0]))
                
                c=list(song_details['title'])[0]
                v=c
                d.append(v)
                e.append(rank_value)
                rank_value+=1
            i += 1
        
    return d,e
    
    

# %%
K=50
urm = data_sparse
MAX_PID = urm.shape[1]
MAX_UID = urm.shape[0]

U, S, Vt = calc_svd(urm, K)
uTest = [li]

uTest_recommended_items = calc_est_matrix(urm, U, S, Vt, uTest, K, True)

d,e=show_rec(uTest)



# %%
# Create DataFrame  
df = pd.DataFrame() 
df['Song'] = d
df['Rank']= e

mod_3=df[:4]


# %%

print("Here is some music for you :")
print("")
print("")

recc = pd.concat([mod_1,mod_2,mod_3])
rec_songs=recc['Song']

result=pd.DataFrame()
result['Songs']=rec_songs

print(result)


