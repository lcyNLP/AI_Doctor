import requests

# 定义请求的URl和传入的数据
url = "http://0.0.0.0:5000/v1/main_serve/"
data = {"uid": "12455", "text": "手臂肌肉酸疼"}

# 向服务发送请求
res = requests.post(url, data=data)
print(res.text)

