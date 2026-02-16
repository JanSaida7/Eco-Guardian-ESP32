import urllib.request
import csv
import ssl

from src.utils import METADATA_URL

METADATA_FILE = "esc50_debug.csv"

def main():
    context = ssl._create_unverified_context()
    print("Downloading metadata...")
    with urllib.request.urlopen(METADATA_URL, context=context) as response, open(METADATA_FILE, 'wb') as out_file:
        out_file.write(response.read())

    print("Categories found:")
    categories = set()
    with open(METADATA_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            categories.add(row['category'])
    
    print(sorted(list(categories)))

if __name__ == "__main__":
    main()
