# Purpose

```
given url
    extract product data
```

# Test

```
source venv/bin/activate
python trainingwheels/classify.py
cd merchantproduct_etl/
python url_to_merchantproduct.py 2>&1 | less
```

