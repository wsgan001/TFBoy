import urllib2

res = urllib2.urlopen('http://blog.leanote.com/qq-alan')
ret = res.read()
print(ret)