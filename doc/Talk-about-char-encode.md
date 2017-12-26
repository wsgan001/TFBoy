### 聊聊字符编码

#### 源起一个Bug
用爬虫在百度爬图片的时候,发现部分查询关键字的时候,出现爬不出图片的情况.比如在爬`鱼`的时候,就没有结果.爬`鱼 图片`就会有结果.

经过异常捕获,发现,在对`URL`转码的时候出现了转码错误:
```python
html = requests.get(url, timeout=10).content.decode('utf-8')

error:
    html = requests.get(url, timeout=10).content.decode('utf-8')
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe7 in position 63897: invalid continuation byte
```

Log的意思是,`UTF-8`的解码器无法处理字符`0xe7`

最后的解决方案是使用`latin-1`编码:
```python
html = requests.get(url, timeout=10).content.decode('latin-1')
```

不过这里遗留了几个问题:
* `0xe7`是什么?
* `latin-1`是什么编码?

下面就要聊一聊这些问题.

#### 计算机编码
在计算机中,所有数据在存储和运算时都要使用二进制表示(这个不理解可以问问冯诺依曼),也就是说,所有字母,符号在计算机中都是由`0`和`1`组成的一串数字表示.但是,就像我们需要给所有事物起名一样,所有现实中的字母,符号都需要一个对应的`0` `1`字串表示,即编码.为了方便大家编码互通,就需要制定统一的编码规则,`ASCII`码就这么产生了.

#### ASCII编码
学编程的时候,`ASCII`是最早介绍的字符编码.
标准`ASCII`使用7位二进制数,因为一个字节占8位,所以在第一位补0形成8位.

举个栗子:
在`ASCII`编码中,字母`A`的表示为:
* 二进制: 0100 0001
* 十进制: 65
* 十六进制: 0x41

所以看前面的问题:`0xe7`是什么?
`0x`是16进制,用二进制表示就是`1110 0111`,十进制是`231`.

对照`ASCII`码,7位一共128个字符,231明显超过了128,所以对于`ASCII`编码来说,它并不认识`0xe7`.
这也说明了一个问题,由于`ASCII`编码长度很短,可以表示的字符有限,遇到中文或者其他字符,就需要其他编码来表示.

比如中文,上万个汉子需要表示,仅用1个字节表示$2^8$个字符是不够的.所以像`GB2312`就是使用两字节表示一个汉字,一共$2^8$ * $2^8$ = 65536个

#### Unicode编码
因为存在不同的编码,所以打开文件前就需要指定正确的编码格式,不然解码出来的都是乱码.
那么能不能出一种编码,能够涵盖所有的字符呢?`Unicode`就是这么样的一个符号集.

但是`Unicode`只是一个符号集,只规定符号二进制代码,没有规定二进制如何存储.

举个栗子:
汉字`鱼`,用`Unicode`表示为十六进制的`9c7c`:
```python
>>> u'鱼'
u'\u9c7c'
```
用二进制表示为`1001 1100 0111 1100`,一共16位,所以至少需要两个字节表示它.

所以问题来了
* 怎么确定它是`Unicode`而不是两个字符组成的`ASCII`?
* 为了解决上面的问题,如果所有字符都用两字节表示,那么只用到7位的字符`A`就会浪费掉将近一个字节的空间.这怎么解决?

#### UTF-8
为了解决空间浪费的问题，出现了一些中间格式的字符集，他们被称为通用转换格式，即UTF（Unicode Transformation Format）。常见的UTF格式有：UTF-7, UTF-7.5, UTF-8,UTF-16, 以及 UTF-32。

主要聊聊常见的`UTF-8`
`UTF-8`规则:
* 如果字符只有一个字节则其最高二进制位为0,后7位是字符的`Unicode`码.单字节的编码和`ASCII`一致
* 对于N字节(N>1),第一个字节前N位设为1,第N+1位为0，其余各字节均以10开头
```bash
2字节: 110xxxxx 10xxxxxx
3字节: 1110xxxx 10xxxxxx 10xxxxxx
4字节: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
```

试着把前面的`鱼`字转成`UTF-8`,`鱼`的二进制16位,正好可以填入3字节的`UTF-8`:`11101001 10110001 10111100`,将其转成16进制就是`e9b1bc`.

> 填入顺序是从低字节开始填,到高字节填不满的补0

用代码验证一下:
```python
>>> u'鱼'.encode('utf-8')
'\xe9\xb1\xbc'
```

了解了`UTF-8`,再回看之前的Bug:`0xe7`的二进制是`1110 0111`,占2字节,对比`UTF-8`的2字节,第三位就不一样,所以`0xe7`不属于`UTF-8`的格式,因此无法解码.

#### ISO/IEC 8859-1
看到这个编码名字是否是一脸懵逼?不过提起别称就清楚了,它就是`Latin-1`编码.

`Latin-1`属于单字节编码,最多能表示0-255的范围,即$2^8$,所以`0xe7`就在它的表示范围内,因此可以解码.

单字节编码的问题是能够表示的字符很少,但是单字节和计算机最基础的表示单位一致,所以面对其他编码的中文表示,可以拆成一个一个的单字节,用`Latin-1`进行保存.所以就像上面`UTF-8`对`鱼`的表示一样,用`Latin-1`解码后,单个字节拼起来就是`UTF-8`编码了:

```python
>>> b'鱼'.decode('Latin-1')
u'\xe9\xb1\xbc'
```

以上就能知道`latin-1`编码可用的原因了.中间的确还有一些没说清楚的问题,等后面遇到问题再继续整理.