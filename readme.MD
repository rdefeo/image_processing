Just `vagrant up`

then to get access to local mongo
`vagrant ssh -- -R 27017:localhost:27017`

Python thing cus of paths
````
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
````