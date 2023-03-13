# Data Processing Info
At the moment one needs to process orderbook, orderflow and volume features separately and then merge them using refactor_data.py.
A single data processing function for processing multiple feature types at the same time is not yet available (work in progress):
- processing orderbook and orderflows requires sequential processing (for rolling window standardization) while 
- processing volumes requires parallel processing (computationally intensive).