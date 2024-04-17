import requests
from bs4 import BeautifulSoup
import pandas as pd
from sqlalchemy import create_engine

# Step 1: Scrape data from the website
url = "http://www.energyonline.com/Data/GenericData.aspx?DataId=19&CAISO___Real-time_Price"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Step 2: Extract table data
table = soup.find("table", {"class": "Gridview"})
rows = table.find_all("tr")

# Initialize lists to store data
data = []

# Extract data from table rows
for row in rows:
    cols = row.find_all("td")
    cols = [ele.text.strip() for ele in cols]
    data.append(cols)

# Remove empty rows
data = [row for row in data if row]

# Step 3: Convert data to DataFrame
columns = ["Time Stamp", "Price"]
df = pd.DataFrame(data[1:], columns=columns)

# Step 4: Store data in SQL repository
engine = create_engine("sqlite:///market_data.db")
df.to_sql("market_data", con=engine, if_exists="replace", index=False)

# Step 5: Save data to an Excel file
df.to_excel("market_data.xlsx", index=False)
