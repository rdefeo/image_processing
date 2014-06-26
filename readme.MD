![Alt text](https://www.codeship.io/projects/e9507f10-c840-0131-f8da-46db0ea8aa49/status)

Just `vagrant up`

then to get access to local mongo
`vagrant ssh -- -R 27017:localhost:27017`

Python thing cus of paths
````
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
````

###Usage

+ install `sudo python setup.py install` from project directory
+ import `from improc.crop import AutoCrop`
+ use `AutoCrop(img)`

###Tests
+ `python2.7 -m unittest discover`
+
