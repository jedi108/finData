from pypoloniex import LoadPairs

# Load realtime data from Poloniex
sess = LoadPairs()

# Returns coin object
LTC = sess.getPair(market = 'BTC', coin = 'LTC')

# Quickview
print (LTC)