import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mymy'))

# sys.path.append('D:\\dev\\python\\finp\\mymy')
print(sys.path)

import mymy.hello
#import mymy.veni
from mymy.veni import Vehicle

mymy.hello.world()

v = Vehicle()
# hello.world()