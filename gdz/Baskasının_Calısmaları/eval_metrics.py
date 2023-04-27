import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def eval_metrics(y_true , y_pred):
    from sklearn.metrics import r2_score , mean_absolute_error , mean_squared_error , mean_absolute_percentage_error
    
    # MAPE hesaplama
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    # r2 hesaplama
    r2 = r2_score(y_true , y_pred)
    
    # mae hesaplama
    mae = mean_absolute_error(y_true , y_pred)

    # rmse hesaplama
    mse = mean_squared_error(y_true,y_pred)**0.5
    
    print(f"""
          Mape Score : {mape}
          R2 Score : {r2}
          MAE Score : {mae}
          MSE Score : {mse}
          """)
    
def eval_plot(y_true , y_pred):
    tests = pd.DataFrame(data = y_true , columns=['Real Values'] , index = X_test[:-24].index)
    preds = pd.DataFrame(data = y_pred , columns=['Predicts'] , index = future_data[:-24].index)
    compare = pd.concat([tests[:-24], preds] , axis= 1)
    print(compare.plot())
    
def eval_df (y_true , y_pred):
    compare = pd.DataFrame({'Real Values': y_true, 'Predicts': y_pred}, index=future_data[:-24].index)
    print(compare)
    
def create_submission(future_preds, num):
    submission_df = pd.DataFrame({'Tarih': future_data.index, 'Dağıtılan Enerji (MWh)': future_preds})
    filename = 'submission{}.csv'.format(num)
    submission_df.to_csv(filename, index=False)
    globals()['submission{}'.format(num)] = submission_df
    
def preds_plot(data, future_data, target_data, anomaly_data=None):
    # Gerçek değerleri mavi renkte çizdir
    plt.plot(data, color='blue', label='Gerçek Değerler')

    # Tahmin edilen değerleri yeşil renkte çizdir
    plt.plot(future_data, color='green', label='Tahminler')

    # Hedef ayın değerleri kırmızı renkte çizdir
    plt.plot(target_data, color='red', label='Hedef Değerler')

    # Anomaly değerleri varsa pembe renkte çizdir
    if anomaly_data is not None:
        plt.plot(anomaly_data, color='black', label='Anormal Değerler')

    # Eksenleri ve grafik başlığını belirle
    plt.title('Gerçek Değerler ve Tahminler')
    plt.xlabel('Saat')
    plt.ylabel('Değer')
    plt.legend()

    # Grafikleri göster
    plt.show()







def get_forecasts(model ):
    # Elimizdeki verilerin son 24'ü bir sonraki tahmini yapmak için kullanılacak
    last_window = X_data[-1:, :, :]

    # 31 gün boyunca gelecek tahminleri yapmak için bir boş tahmin dizisi oluşturun
    forecasts = []

    for i in range(31*24):
        # Tahmin edilen değerleri ölçeklendirmek için son pencereyi yeniden şekillendirin
        last_window_reshaped = last_window.reshape(1, 24, 20)

        # Son pencereyi kullanarak bir tahmin yapın
        forecast = model.predict(last_window_reshaped , verbose=0)[0][0]

        # Tahmini tahmin listesine ekleyin
        forecasts.append(forecast)
        
        # future_data_scaled'ı güncelle
        future_data_scaled[i,0] = forecast

        # Tahmin edilen değeri son pencerenin sonuna ekleyin
        last_window = np.append(last_window[:,1:,:] , np.array([[future_data_scaled[i]]]) , axis = 1)

        
    # Tahminleri geri ölçeklendirin
    forecast_data = scaler.inverse_transform(np.array(future_data_scaled).reshape(-1, 20))
    forecasts = forecast_data[:,0]
    
    return forecasts




def compareAll(y_true , y_forecast comparisions = 'all'):
    def compare_rmse(y_true , y_forecast):
        # Root Mean Square Error (RMSE)
        rmse = np.sqrt(np.mean((np.array(y_true) - np.array(y_forecast))**2))
        return rmse

    def compare_mae(y_true , y_forecast):
        mae = np.mean(np.abs(np.array(y_true) - np.array(y_forecast)))
        return mae
        
    def compare_corr(y_true , y_forecast):
        from scipy.stats import pearsonr
        corr, _ = pearsonr(y_true, y_forecast)
        return corr

    def compare_cosine_similarity(y_true, y_pred):
        """
        Calculates the cosine similarity between two lists.
        """
        dot_product = np.dot(y_true, y_pred)
        norm_list1 = np.linalg.norm(y_true)
        norm_list2 = np.linalg.norm(y_pred)
        return dot_product / (norm_list1 * norm_list2)
        
    def compare_jaccard_similarity(y_true, list2):
        """
        Calculates the Jaccard similarity between two lists.
        """
        set1 = set(y_true)
        set2 = set(list2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union)
    
    rmse = compare_rmse(y_true,y_forecast)
    mae = compare_mae(y_true,y_forecast)
    corr = compare_corr(y_true,y_forecast)
    cos_sim = compare_cosine_similarity(y_true,y_forecast)
    jac_sim = compare_jaccard_similarity(y_true,y_forecast)
    
    print(f"""
          RMSE : {rmse}
          MAE : {mae}
          CORRELATION : {corr}
          COSINUS SIMILARITY : {cos_sim}
          JACCARD SIMILARITY : {jac_sim}
          """)
    
    
    
    
    


def compare_rmse(y_true , y_forecast):
    # Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean((np.array(y_true) - np.array(y_forecast))**2))
    return rmse

def compare_mae(y_true , y_forecast):
    mae = np.mean(np.abs(np.array(y_true) - np.array(y_forecast)))
    return mae
    
def compare_corr(y_true , y_forecast):
    from scipy.stats import pearsonr
    corr, _ = pearsonr(y_true, y_forecast)
    return corr

def compare_cosine_similarity(y_true, y_pred):
    """
    Calculates the cosine similarity between two lists.
    """
    dot_product = np.dot(y_true, y_pred)
    norm_list1 = np.linalg.norm(y_true)
    norm_list2 = np.linalg.norm(y_pred)
    return dot_product / (norm_list1 * norm_list2)
    
def jaccard_similarity(y_true, list2):
    """
    Calculates the Jaccard similarity between two lists.
    """
    set1 = set(y_true)
    set2 = set(list2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)