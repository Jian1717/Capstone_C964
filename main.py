import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVR


def add_charging_station_density(combined_df, df_ev_charging_station):
    radius_miles = 10
    radius_radians = radius_miles / 3958.8

    # Fit NearestNeighbors to charging stations
    nbrs = NearestNeighbors(radius=radius_radians, metric='haversine')
    charging_coordinates = np.radians(df_ev_charging_station[['Latitude', 'Longitude']].values)
    nbrs.fit(charging_coordinates)

    # Count the number of nearby charging stations for each EV
    ev_coordinates = np.radians(combined_df[['Latitude', 'Longitude']].values)
    counts = nbrs.radius_neighbors(ev_coordinates, return_distance=False)

    # Add the counts to combined_df
    combined_df['Nearby_Charging_Stations_Count'] = [len(count) for count in counts]


def add_closest_charging_station(combined_df, df_ev_charging_station):
    nbrs = NearestNeighbors(n_neighbors=1, metric='haversine')
    charging_coordinates = df_ev_charging_station[['Latitude', 'Longitude']].values
    nbrs.fit(np.radians(charging_coordinates))
    ev_coordinates = combined_df[['Latitude', 'Longitude']].values
    # Find the nearest charging stations for each EV location
    distances, indices = nbrs.kneighbors(np.radians(ev_coordinates))
    distances_miles = distances.flatten() * 3958.8
    combined_df['Target_Charging_Station_Latitude'] = df_ev_charging_station.iloc[indices.flatten()]['Latitude'].values
    combined_df['Target_Charging_Station_Longitude'] = df_ev_charging_station.iloc[indices.flatten()][
        'Longitude'].values
    combined_df['Nearest_Charging_Station'] = df_ev_charging_station.iloc[indices.flatten()]['City'].values
    combined_df['Distance_to_Nearest_Charging_Station'] = distances_miles


def read_data(file_path, file_path_1):
    name = ['City', 'State', 'Postal Code', 'Vehicle Location']
    df_registered_ev = pd.read_csv(file_path, usecols=name)
    # Extract Longitude and Latitude from 'Vehicle Location'
    df_registered_ev[['Longitude', 'Latitude']] = df_registered_ev['Vehicle Location'].str.extract(
        r'POINT \(([^ ]+) 'r'([^ ]+)\)')
    df_registered_ev = df_registered_ev.dropna(subset=['Vehicle Location'])
    df_registered_ev = df_registered_ev.drop(columns=['Vehicle Location'])
    df_registered_ev.drop(df_registered_ev[df_registered_ev['State'] != 'WA'].index, inplace=True)
    df_registered_ev['Postal Code'] = df_registered_ev['Postal Code'].astype(int)
    ev_count = df_registered_ev.groupby(['City', 'State', 'Postal Code']).size().reset_index(name='EV_Count')
    name = ['City', 'State', 'Postal Code', 'Charging Station Location', 'Number Of Charge Stand']
    df_ev_charging_station = pd.read_csv(file_path_1, usecols=name,
                                         dtype={'Postal Code': int, 'Number Of Charge Stand': int})
    # Extract Longitude and Latitude from 'Charging Station Location'
    df_ev_charging_station[['Latitude', 'Longitude']] = df_ev_charging_station['Charging Station Location'].str.extract(
        r'([\d.-]+), ([\d.-]+)')
    df_ev_charging_station = df_ev_charging_station.drop(columns=['Charging Station Location'])
    ev_charging_station_count = df_ev_charging_station.groupby(['City', 'State', 'Postal Code']).size().reset_index(
        name='EV_Charging_Station_Count')
    ev_charging_stand_count = df_ev_charging_station.groupby(['City', 'State', 'Postal Code'])[
        'Number Of Charge Stand'].sum().reset_index(
        name='EV_Charging_Stand_Count')
    # Merge EV Count
    combined_df = pd.merge(df_registered_ev, ev_count, on=['City', 'State', 'Postal Code'], how='left')

    # Merge Charging Station Count
    combined_df = pd.merge(combined_df, ev_charging_station_count, on=['City', 'State', 'Postal Code'], how='left')

    # Merge Total Charging Stands Count
    combined_df = pd.merge(combined_df, ev_charging_stand_count, on=['City', 'State', 'Postal Code'], how='left')

    combined_df.fillna({'EV_Count': 0, 'EV_Charging_Station_Count': 0, 'EV_Charging_Stand_Count': 0}, inplace=True)

    combined_df['EV_Charging_Stand_Count'] = combined_df['EV_Charging_Stand_Count'].astype(int)

    combined_df['EV_Charging_Station_Count'] = combined_df['EV_Charging_Station_Count'].astype(int)

    combined_df['Latitude'] = pd.to_numeric(combined_df['Latitude'], errors='coerce')
    combined_df['Longitude'] = pd.to_numeric(combined_df['Longitude'], errors='coerce')
    df_ev_charging_station['Latitude'] = pd.to_numeric(df_ev_charging_station['Latitude'], errors='coerce')
    df_ev_charging_station['Longitude'] = pd.to_numeric(df_ev_charging_station['Longitude'], errors='coerce')

    add_charging_station_density(combined_df, df_ev_charging_station)
    add_closest_charging_station(combined_df, df_ev_charging_station)
    data_set_file_path = 'Data_Set.csv'
    combined_df.to_csv(data_set_file_path, index=False)

    return combined_df


if __name__ == '__main__':
    registered_ev_file_path = 'Electric_Vehicle_Population_Data.csv'
    ev_charging_station_file_path = 'EV_Charging_Station_Data.csv'
    combined_df = read_data(registered_ev_file_path, ev_charging_station_file_path)
    X = combined_df[['City', 'State', 'Postal Code', 'Latitude', 'Longitude', 'EV_Count', 'EV_Charging_Stand_Count',
                     'EV_Charging_Station_Count', 'Nearby_Charging_Stations_Count', 'Nearest_Charging_Station',
                     'Distance_to_Nearest_Charging_Station']]
    y = combined_df[['Target_Charging_Station_Latitude', 'Target_Charging_Station_Longitude']]
    # One-hot encoding for categorical variables
    categorical_features = ['City', 'State', 'Nearest_Charging_Station']
    numeric_features = X.columns.difference(categorical_features)

    # Create a preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),  # Scale numeric features
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # Apply one-hot encoding
        ])

    # Create a pipeline that first preprocesses the data and then fits the model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', MultiOutputRegressor(SVR(kernel='rbf')))  # Using RBF kernel for SVR
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Calculate performance metrics
    mae_latitude = mean_absolute_error(y_test.iloc[:, 0], y_pred[:, 0])
    mse_latitude = mean_squared_error(y_test.iloc[:, 0], y_pred[:, 0])

    mae_longitude = mean_absolute_error(y_test.iloc[:, 1], y_pred[:, 1])
    mse_longitude = mean_squared_error(y_test.iloc[:, 1], y_pred[:, 1])

    print(f'Latitude - Mean Absolute Error: {mae_latitude}, Mean Squared Error: {mse_latitude}')
    print(f'Longitude - Mean Absolute Error: {mae_longitude}, Mean Squared Error: {mse_longitude}')