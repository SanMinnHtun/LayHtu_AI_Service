import requests
import pandas as pd
import time

API_KEY = "9613be12dc59a63ec516c5b2fc9323dc"
LAT = 16.8661  # Yangon
LON = 96.1951

# Calculate Timestamps (3 Months = 90 Days)
end_time = int(time.time())
start_time = end_time - (90 * 24 * 60 * 60)

def fetch_and_save():
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={LAT}&lon={LON}&start={start_time}&end={end_time}&appid={API_KEY}"

    print("⏳ Fetching data from OpenWeather... Please wait.")
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        all_records = []

        for item in data['list']:
            record = {
                'timestamp': item['dt'],
                'aqi': item['main']['aqi'],
                'co': item['components']['co'],
                'no2': item['components']['no2'],
                'o3': item['components']['o3'],
                'pm2_5': item['components']['pm2_5'],
                'pm10': item['components']['pm10']
            }
            all_records.append(record)

        df = pd.DataFrame(all_records)
        df.to_csv('myanmar_air_data.csv', index=False)
        print(f"✅ Success! Saved {len(df)} rows to 'myanmar_air_data.csv'")
    else:
        print(f"❌ Error: {response.status_code}. check if your key is active!")

if __name__ == "__main__":
    fetch_and_save()