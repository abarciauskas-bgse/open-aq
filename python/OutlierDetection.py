# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import DBSCAN
import statsmodels.api as sm
from uszipcode import ZipcodeSearchEngine

class Outlier_Detection(object):
    
    #TODO: kwargs and params for methods 
    
    def __init__(self, us_data_pm25_df):

        self.us_data_pm25_df = us_data_pm25_df

    def extract_time( self ):
        
        '''
        This function extracts the hour and the day number in the month from
        the local datetime
        
        It consumes a dataframe that contains local datetime
        
        It returns:
            - a dataframe of unique datetimes and their mapping to 
                hour and day number
            - the input dataframe with hour and day number columns merged
        '''
        
        us_data_pm25_df = self.us_data_pm25_df
        
        date_map_df = pd.DataFrame(us_data_pm25_df["local"].drop_duplicates())

        # creating a datetime format
        date_map_df["local_datetime"] = pd.to_datetime(date_map_df["local"])

        # extracting the time and day
        date_map_df["local_hour"] = date_map_df["local_datetime"].dt.hour
        date_map_df["local_day"] = date_map_df["local_datetime"].dt.day
        
        # extract the hour and day from the local time
        us_data_pm25_df = us_data_pm25_df.merge( date_map_df, on="local", how="left" )

        self.date_map_df, self.us_data_pm25_df = date_map_df, us_data_pm25_df
    

    def coords_to_df( self ): 
        
        '''
        This function consumes a dataframe that contains:
            - location name
            - latitude
            - longitude
            
        It then removes duplicates and creates a matrix to be consumed
        by the kmeans model
        '''
        
        us_data_pm25_df = self.us_data_pm25_df
        # Create a DataFrame of Unique Location Coordinates of ALL sensors in the US
        coords_df = us_data_pm25_df[['location','latitude', 'longitude']].drop_duplicates()\
                                                                    .reset_index()\
                                                                    .drop("index",axis=1)
        # Prep the Coordinate DF for a distance matrix
        coords = coords_df.as_matrix(columns=['latitude', 'longitude'])

        self.coords_df, self.coords = coords_df, coords


    def kmeans_gsearch( self ):
        
        '''
        This function consumes the unique coordinates and iterates over kmeans models
        
        While it iterates, it saves a list of means for the clusters in each model for:
            - silhouette score
            - sum of squared errors
            
        The silhouette score is a ratio of the compactness and isolation for the clusters
        
        We will choose the number of clusters by finding the highest silhouette score
        '''
        
        coords_df = self.coords_df
        
        # create feature vectors to train a kmeans model
        locations = coords_df['location'].values
        f1 = coords_df['latitude'].values
        f2 = coords_df['longitude'].values
        X = np.array(list(zip(f1, f2)))

        # Looping through silhouette scores to find an optimal compactness and isolation
        # TODO: Revisit this range. It was chosen hueristically 
        cluster_range = range( 30, 85 )
        cluster_errors = []
        cluster_sil_means = []

        for num_clusters in cluster_range:
            clusters = KMeans( num_clusters, random_state=0 )
            clusters.fit( X )

            cluster_errors.append( clusters.inertia_ )

            cluster_labels = clusters.fit_predict( X )

            silhouette_avg = silhouette_score( X, cluster_labels )
            cluster_sil_means.append( silhouette_avg )

        clusters_df = pd.DataFrame( {\
                                       "num_clusters":cluster_range\
                                     , "cluster_errors": cluster_errors
                                     , "mean_sil": cluster_sil_means } )

        self.clusters_df,self.cluster_errors,self.cluster_sil_means,self.X =\
    clusters_df,cluster_errors,cluster_sil_means,X

    def kmeans_optimized( self ):
        
        '''
        This function runs the kmeans model with the optimal number of clusters, chosen 
        with the max silhouette score
        '''
        
        clusters_df = self.clusters_df
        X = self.X 

        sil_info = clusters_df[clusters_df.mean_sil==clusters_df.mean_sil.max()]
        k = sil_info.num_clusters.values[0]

        # Number of clusters
        kmeans = KMeans(n_clusters=k, random_state=0)
        # Fitting the input data
        kmeans = kmeans.fit(X)
        # Getting the cluster labels
        labels = kmeans.predict(X)
        # Centroid values
        centroids = kmeans.cluster_centers_

        self.labels, self.sil_info, self.k = labels, sil_info, k

    def merge_labels( self ):
        
        '''
        This function merges the cluster labels from the kmeans model to the 
        time series dataset 
        '''
        
        coords_df = self.coords_df
        us_data_pm25_df = self.us_data_pm25_df
        labels = self.labels
        
        coords_df["geo_cluster"] = labels

        # Merge Geo Clusters in on lat lng for precision
        working_df = us_data_pm25_df.merge(\
                                             coords_df[["latitude","longitude","location","geo_cluster"]]\
                                           , on=["latitude","longitude","location"], how="left" )

        self.working_df, self.coords_df = working_df, coords_df

    def dist_matrix( self ):
        
        '''
        This function creates a pairwise distance matrix for the cluster centroids
        based on latitude and longitude, then converts it to km
        '''
        
        working_df = self.working_df

        # Calculate the mid point
        midpoints_df = working_df[[\
                                     "geo_cluster"\
                                   , "latitude"\
                                   , "longitude"\
                                  ]].groupby("geo_cluster")\
                                    .mean()\
                                    .reset_index()

        # Calculate a distance matrix
        dist_matrix = pd.DataFrame(\
                                     squareform(pdist( midpoints_df.iloc[:, 1:] ))\
                                   , columns= midpoints_df.geo_cluster.unique()\
                                   , index= midpoints_df.geo_cluster.unique() )

        # 1 degree is about 111km
        # TODO: This is actually variable
        dist_matrix_km = dist_matrix*111

        self.dist_matrix_km, self.midpoints_df = dist_matrix_km, midpoints_df

    def flag_neighbor(self, x, km=400):
        
        # A default parameter is set to 400km
        # TODO: Revisit this distance. This was just a guess for how far PM 2.5 effects could reach. There's no support.
        if (int(x) !=0) & (int(x) <=km):
            return 1
        else:
            return 0

    def neighbor_lookup( self ):
        
        '''
        This function uses the distance matrix to create a list of neighboring centroid within 400km for 
        each cluster centroid
            - The output is a dictionary
        '''

        dist_matrix_dummies = self.dist_matrix_km.applymap(self.flag_neighbor)\
                                            .reset_index()\
                                            .rename(columns={"index":"geo_cluster"})
        
        # Create a dictionary that contains a list of neighbors for each cluster
        # We can pass this in as a filter to make it easier to map model fit logic later
        neighbor_dict = {}

        # Loop through the unique clusters
        for cluster in dist_matrix_dummies["geo_cluster"].unique():
            # Transpose the row into a column vector
            neighbor_flags = dist_matrix_dummies[dist_matrix_dummies["geo_cluster"]==cluster].iloc[:, 1:].T
            neighbor_flags = neighbor_flags.rename(\
                                                   columns={\
                                                              neighbor_flags.columns.values[0]:"neighbor_ind"\
                                                            })\


            # When there is a neighbor flag, take the index and store it in a dictionary
            neighbor_list = list(neighbor_flags[neighbor_flags["neighbor_ind"]==1].index)
            neighbor_dict[cluster] = neighbor_list

        self.neighbor_dict = neighbor_dict

    def get_cities( self ):
        
        '''
        This function uses the zipcodes API to return the city name for each cluster centroid,
        based on latitude and longitude
        '''
        
        midpoints_df = self.midpoints_df
        
        search = ZipcodeSearchEngine()
        midpoints_df["City"] = midpoints_df[["latitude", "longitude"]].apply(lambda x:\
                                                      search.by_coordinate(\
                                                                              x[0]\
                                                                           , x[1]\
                                                                           , radius=30\
                                                                           , returns=1)[0].City\
                                                      , axis=1)

        midpoints_df["State"] = midpoints_df[["latitude", "longitude"]].apply(lambda x:\
                                                      search.by_coordinate(\
                                                                              x[0]\
                                                                           , x[1]\
                                                                           , radius=30\
                                                                           , returns=1)[0].State\
                                                      , axis=1)

        midpoints_df["City_State"] =midpoints_df["City"]+", "+midpoints_df["State"]
        cities_dict = midpoints_df.set_index("geo_cluster").to_dict("index")
        self.cities_dict = cities_dict

    def resample_and_unstack( self ):
        
        '''
        This function chooses a single hour of the day and dedups the time series, then
        creates a column for each cluster's mean reading values for PM 2.5
        '''
        
        working_df = self.working_df
        
        grouped_ts_df = working_df[working_df["local_hour"]==18]\
                                    [["geo_cluster","location","local_datetime","value"]]\
                                        .groupby(["geo_cluster", "local_datetime"])\
                                        .mean()\
                                        .reset_index()

        df = grouped_ts_df[["geo_cluster","local_datetime","value"]].drop_duplicates()\
                                                                    .copy()

        # set a datetime64 dtype to the date, so that we can use it as an index    
        df.local_datetime = pd.to_datetime(df.local_datetime)

        # set the index as the date field
        df = df.set_index('local_datetime') 

        # create a generator function for groupby operation, that resamples to a 1 day
        ## for the entire dataset, and then unstack it into a column for each cluster
        grouper = df.groupby([pd.Grouper(freq='1D'), "geo_cluster"])
        unstacked_df = grouper['value'].sum().unstack('geo_cluster')
        self.unstacked_df = unstacked_df

    def non_persistent_ts( self ):
        
        '''
        This function looks at a rolling 4 records window and creates an indicator
        if any records are null
        
        If more than 20 percent of the records have an indicator, then the cluster
        is dropped
        '''
        
        unstacked_df = self.unstacked_df
        
        window_null_ct_df = unstacked_df.apply( lambda x: x.isnull() )\
                                        .rolling(4)\
                                        .sum()

        f = lambda x: 1 if x==4 else 0
        window_null_ct_df = window_null_ct_df.applymap( f ).reset_index()

        window_null_ct_df["year"] = window_null_ct_df["local_datetime"].map( lambda x: x.year )

        year_ct = window_null_ct_df.groupby("year")\
                         .count()

        year_sum = window_null_ct_df.groupby("year")\
                          .sum()

        window_null_rates = (year_sum/year_ct).T[:-1].reset_index()

        drop_list = list(window_null_rates[window_null_rates[2016]>.2]["geo_cluster"])
        keep_list = [i for i in unstacked_df.columns if i not in drop_list]

        self.keep_list, self.drop_list, self.window_null_rates = keep_list, drop_list, window_null_rates

    def treat_ts( self ):
        
        '''
        This function interpolates between null records with a cubic spline
        '''
        
        keep_list = self.keep_list
        unstacked_df = self.unstacked_df
        
        # drop the bad columns
        unstacked_df = unstacked_df[keep_list]

        # interpolate the missing records
        self.treated_ts_df = unstacked_df.loc[:,0:].apply(lambda x: x.interpolate(method='spline', order=3))

    def apply_lowess(self, x):
        
        '''
        This function applies lowess smoothing to each centroid column
        '''
        
        treated_ts_df = self.treated_ts_df
        
        lowess = sm.nonparametric.lowess

        sm_lowess_df = x.dropna()

        dates_df=pd.DataFrame(treated_ts_df.index)

        x=sm_lowess_df.index
        y=sm_lowess_df

        w = lowess(y, x, frac=1./3)

        f = lambda x: x[0]
        output_df = pd.DataFrame(map(f,w[:,1:]), columns=["LOWESS"])
        output_df["local_datetime"]=x

        merged_df=dates_df.merge(output_df, on="local_datetime", how="left")

        return merged_df["LOWESS"]

    
    def calc_sq_residuals( self ):  
        
        '''
        This function subtracts each cluster's lowess smoothed time series from the 
        time series and then squares the residuals 
        '''
        
        treated_ts_df = self.treated_ts_df

        # lowess smoother
        smoother_applied_df = treated_ts_df.apply(lambda x: self.apply_lowess(x)).set_index(treated_ts_df.index)

        # residuals 
        calc_resids_applied_df = treated_ts_df-smoother_applied_df
        calc_sq_resids_applied_df = calc_resids_applied_df**2

        self.calc_sq_resids_applied_df, self.smoother_applied_df = calc_sq_resids_applied_df, smoother_applied_df

    # TODO: grid seach epsilons according to volatility of each cluster
    def cluster_residuals( self, X ): 
        
        '''
        This function maps DBSCAN to the squared residuals for each cluster and identify outliers for 
        each time series 
        '''
        
        db = DBSCAN(\
                      eps=.9\
                    , min_samples=4\
                    , leaf_size=30\
                    , metric="euclidean"\
                   ).fit(X)

        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        return db.labels_
    
    def get_outliers( self ):
        
        '''
        This function creates a mask dataframe for the outlier indicator and applies it to the 
        original time series, which results in the outlier PM 2.5 measurement value or a null 
        '''
        
        calc_sq_resids_applied_df = self.calc_sq_resids_applied_df
        treated_ts_df = self.treated_ts_df
        
        # map the density based clustering to squared residuals for each cluster
        f = lambda x: self.cluster_residuals( np.array(x.fillna(0)).reshape(-1, 1) )
        flagged_df = calc_sq_resids_applied_df.apply(f, axis=0).apply(lambda x: (x==-1)*1 ).replace(0,np.nan)

        # multiply the indicators for each smoothing method by the original time series for the target
        ### this will give us the outlier value to plot
        ### there's a better way to do this for sure
        outliers_ts_df = flagged_df*treated_ts_df 
        
        self.outliers_ts_df, self.flagged_df = outliers_ts_df, flagged_df