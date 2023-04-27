import numpy as np
import holidays
import pandas as pd
import matplotlib.pyplot as plt

# def create_datetimes(df, type='normal', label=None):

#     df['date'] = df.index
        
#     df['hour'] = df['date'].dt.hour
#     df['dayofweek'] = df['date'].dt.dayofweek
#     df['quarter'] = df['date'].dt.quarter
#     df['month'] = df['date'].dt.month
#     df['year'] = df['date'].dt.year
#     df['dayofyear'] = df['date'].dt.dayofyear
#     df['dayofmonth'] = df['date'].dt.day
#     df['weekofyear'] = df['date'].dt.isocalendar().week
    
    
#     if type == 'dummy':
#             cat_cols = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']
#             dummy_df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
#             df = pd.concat([df, dummy_df], axis=1)
#             df = df.drop(cat_cols, axis=1)
    
#     return df


def create_time_features1(df, med, calendar , submit):
    global merged_df , merged_df2
    
    med["Yıl"] = pd.to_datetime(med["Tarih"]).dt.year
    med["Ay"] = pd.to_datetime(med["Tarih"]).dt.month
    med["Gün"] = pd.to_datetime(med["Tarih"]).dt.day

    df["Yıl"] = pd.to_datetime(df["Tarih"]).dt.year
    df["Ay"] = pd.to_datetime(df["Tarih"]).dt.month
    df["Gün"] = pd.to_datetime(df["Tarih"]).dt.day
    df["Saat"] = pd.to_datetime(df["Tarih"]).dt.hour

    submit["Yıl"] = pd.to_datetime(submit["Tarih"]).dt.year
    submit["Ay"] = pd.to_datetime(submit["Tarih"]).dt.month
    submit["Gün"] = pd.to_datetime(submit["Tarih"]).dt.day
    submit["Saat"] = pd.to_datetime(submit["Tarih"]).dt.hour
    submit["Kesintili Günler"]=0
    
    
    for idx, row in df.iterrows():
        mask = (med["Yıl"] == row["Yıl"]) & (med["Ay"] == row["Ay"]) & (med["Gün"] == row["Gün"])
        if med.loc[mask].empty:
            df.loc[idx, "Kesintili Günler"] = 0
        else:
            df.loc[idx, "Kesintili Günler"] = 1
        

    
    

    df['Yıl'] = df['Tarih'].dt.year
    df['Ay'] = df['Tarih'].dt.month
    df['Gün'] = df['Tarih'].dt.day
    df['Quarter'] = df['Tarih'].dt.quarter
    df['day_of_week'] = df['Tarih'].dt.dayofweek
    df['day_of_year'] = df['Tarih'].dt.dayofyear
    df['Saat'] = df['Tarih'].dt.hour
    df['week_of_year'] = df['Tarih'].dt.isocalendar().week

    new_df = calendar.iloc[853:2557].copy()
    new_df.loc[:, "Ay"] = pd.to_datetime(new_df["CALENDAR_DATE"], format="%d.%m.%Y").dt.month
    new_df.loc[:, "Yıl"] = pd.to_datetime(new_df["CALENDAR_DATE"], format="%d.%m.%Y").dt.year
    new_df = new_df.drop(["SEASON_SK", "SPECIAL_DAY_SK", "SPECIAL_DAY_SK2"], axis=1)

    new_df["WEEKEND_FLAG"] =new_df["WEEKEND_FLAG"].replace(["N"],0)
    new_df["WEEKEND_FLAG"] =new_df["WEEKEND_FLAG"].replace(["Y"],1)
    new_df["RAMADAN_FLAG"] =new_df["RAMADAN_FLAG"].replace(["N"],0)
    new_df["RAMADAN_FLAG"] =new_df["RAMADAN_FLAG"].replace(["Y"],1)
    new_df["PUBLIC_HOLIDAY_FLAG"] =new_df["PUBLIC_HOLIDAY_FLAG"].replace(["N"],0)
    new_df["PUBLIC_HOLIDAY_FLAG"] =new_df["PUBLIC_HOLIDAY_FLAG"].replace(["Y"],1)

    new_df.rename(columns={'DAY_OF_MONTH': 'Gün'}, inplace=True)
    new_df.rename(columns={'DAY_OF_WEEK_SK': 'Haftanın Günü'}, inplace=True)
    new_df.rename(columns={'QUARTER_OF_YEAR': 'Sezon'}, inplace=True)
    new_df.rename(columns={'WEEKEND_FLAG': 'Haftasonu - Haftaiçi'}, inplace=True)
    new_df.rename(columns={'WEEK_OF_YEAR': 'Yılın kaçıncı haftası'}, inplace=True)
    new_df.rename(columns={'RAMADAN_FLAG': 'Ramazan'}, inplace=True)
    new_df.rename(columns={'RELIGIOUS_DAY_FLAG_SK': 'Dini Gün'}, inplace=True)
    new_df.rename(columns={'NATIONAL_DAY_FLAG_SK': 'Ulusal Gün'}, inplace=True)
    new_df.rename(columns={'PUBLIC_HOLIDAY_FLAG': 'Resmi tatil'}, inplace=True)

    new_df_submit=new_df.iloc[0:31]
    new_df_train=new_df.iloc[31:1704]
    merged_df = pd.merge(df, new_df_train, on=["Yıl", "Ay", "Gün"])
    merged_df2 = pd.merge(submit, new_df_submit, on=["Yıl", "Ay", "Gün"])
    

    # Add other time features here as needed
    # For example: weekofyear, quarter, is_weekend, is_holiday, etc.
    
    return merged_df , merged_df2 




def create_time_features2(df , med):
    
    df['Yıl'] = df['Tarih'].dt.year
    df['Ay'] = df['Tarih'].dt.month
    df['Gün'] = df['Tarih'].dt.day
    df['Quarter'] = df['Tarih'].dt.quarter
    df['day_of_week'] = df['Tarih'].dt.dayofweek
    df['day_of_year'] = df['Tarih'].dt.dayofyear
    df['hour'] = df['Tarih'].dt.hour
    df['week_of_year'] = df['Tarih'].dt.isocalendar().week
    
    
    
    
    turkey_holidays = holidays.Turkey()

    def is_holiday(date): 
        return date in turkey_holidays

    df['holiday'] = pd.Series(df['Tarih'].dt.date).apply(is_holiday).astype(int)
    df['weekend'] = df['day_of_week'].apply(lambda x: 1 if x in (5, 6) else 0)
    
    
    outage_dates = set(med['Tarih'].dt.date)
    df['electrical_outage'] = df['Tarih'].dt.date.apply(lambda x: 1 if x in outage_dates else 0)
    
    
    conditions = [
    (6 <= df['hour']) & (df['hour'] < 12),
    (12 <= df['hour']) & (df['hour'] < 18),
    (18 <= df['hour']) & (df['hour'] < 24)
    ]
    choices = [1, 2, 3]

    df['time_of_day'] = np.select(conditions, choices, default=3)
    df['time_of_day'] = df['time_of_day'].astype('int')
    

    df['business_day'] = df['day_of_week'].apply(lambda x: 1 if x in range(0, 5) else 0)
    df['cumulative_holidays'] = df['holiday'].cumsum()
        
        
    df['outage_percentage'] = (df['electrical_outage'].cumsum() / pd.Series(range(1, len(df) + 1))).astype(float) * 100


    window_size = 24
    df['rolling_outages_24h'] = df['electrical_outage'].rolling(window=window_size).sum()
    # df['rolling_outages_24h'].iloc[:window_size-1] = df['rolling_outages_24h'].iloc[window_size-1]
    df.loc[:window_size-1, 'rolling_outages_24h'] = df['rolling_outages_24h'].iloc[window_size-1]


    
    
    alpha = 0.1
    df['exp_avg_outages_24h'] = df['electrical_outage'].ewm(alpha=alpha).mean()
        
        

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
    
    def is_spring(ds):
        date = pd.to_datetime(ds)
        if (date.month >= 3) & (date.month <= 5):
            return 1
        else :
            return 0

    def is_summer(ds):
        date = pd.to_datetime(ds)
        if (date.month >= 6) & (date.month <= 8):
            return 1
        else :
            return 0

    def is_autumn(ds):
        date = pd.to_datetime(ds)
        if (date.month >= 9) & (date.month <= 11):
            return 1
        else :
            return 0

    def is_winter(ds):
        date = pd.to_datetime(ds)
        if (date.month >= 12) | (date.month <= 2):
            return 1
        else :
            return 0

    def is_weekend(ds):
        date = pd.to_datetime(ds)
        if date.weekday() in (5,6):
            return 1
        else :
            return 0



    # adding to train set
    df['is_spring'] = df['Tarih'].apply(is_spring)
    df['is_summer'] = df['Tarih'].apply(is_summer)
    df['is_autumn'] = df['Tarih'].apply(is_autumn)
    df['is_winter'] = df['Tarih'].apply(is_winter)
    df['is_weekend'] = df['Tarih'].apply(is_weekend)
    df['is_weekday'] = ~df['Tarih'].apply(is_weekend)

    # # adding to test set
    # df['is_spring'] = df['Tarih'].apply(is_spring)
    # df['is_autumn'] = df['Tarih'].apply(is_autumn)
    # df['is_winter'] = df['Tarih'].apply(is_winter)
    # df['is_weekend'] = df['Tarih'].apply(is_weekend)
    # df['is_weekday'] = ~df['Tarih'].apply(is_weekend)
        
        
    return df
        
    














    
    

    

    


    
    


    
    


    


# def create_isramadan(df):
#     hols = pd.read_csv('Calendar.csv', parse_dates=['CALENDAR_DATE'])
#     hols = hols[['CALENDAR_DATE','RAMADAN_FLAG','PUBLIC_HOLIDAY_FLAG']]
#     df['isRamadan'] = np.where((hol['RAMADAN_FLAG'] == 'Y') | (hol['PUBLIC_HOLIDAY_FLAG'] == 'Y'), 'TR-Holidays', 0)
